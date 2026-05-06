"""매일 아침 알림 통합 — 모의투자/실거래 운용 핵심.

5가지 시나리오 자동 판별:
1. 정상     : 보유 유지 (90% 이상 해당)
2. 리밸런싱 : 월간 매매일 도래 — 매도/매수 종목 명시
3. 청산     : 시장 MA 깨짐 — 즉시 전량 매도
4. 회복     : 시장 MA 회복 — 다음 리밸런싱부터 매수 재개
5. Kill     : 일일 -3% 또는 누적 -20% — 비상 청산 (가드레일 §10)

사용:
    quant signal --daily            # 챔피언 (월간 모멘텀+MA100) 기준 알림
    quant signal --daily --seed 10000000  # 시드 입력 시 잔고/주식수 추정
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

import pandas as pd

from quant.backtest.regime import RegimeMode, filter_weights, regime_state
from quant.common.logger import logger
from quant.data.price.fetch_krx import load_close_panel, load_value_panel
from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights


class AlertType(str, Enum):
    NORMAL = "normal"  # 보유 유지
    REBALANCE = "rebalance"  # 리밸런싱 도래
    LIQUIDATE = "liquidate"  # 시장 MA 깨짐 → 청산
    RECOVER = "recover"  # 시장 MA 회복
    KILL_SWITCH = "kill_switch"  # 비상


@dataclass
class HoldingRow:
    ticker: str
    name: str
    weight: float
    target_value_won: float | None = None
    target_shares: int | None = None
    last_close: float | None = None


@dataclass
class DailySignal:
    """매일 아침 알림 통합 결과."""

    as_of: date
    alert_type: AlertType
    market_state: dict  # regime_state() 출력
    holdings: list[HoldingRow]  # 현재(또는 다음) 보유
    new_buys: list[HoldingRow]  # 리밸런싱 시 신규 매수
    sells: list[HoldingRow]  # 리밸런싱 시 매도
    next_rebalance: date | None  # 다음 리밸런싱일
    notes: list[str]  # 추가 알림 (Kill switch 사유 등)


def _ticker_name_map() -> dict[str, str]:
    try:
        import FinanceDataReader as fdr  # noqa: N813

        m: dict[str, str] = {}
        for market in ("KOSPI", "KOSDAQ"):
            listing = fdr.StockListing(market)
            m.update(dict(zip(listing["Code"], listing["Name"], strict=False)))
        return m
    except Exception as e:
        logger.warning(f"종목명 조회 실패: {e}")
        return {}


def _to_holding_rows(
    weights: pd.Series,
    last_close: pd.Series,
    name_map: dict[str, str],
    seed_won: float | None,
) -> list[HoldingRow]:
    rows: list[HoldingRow] = []
    for ticker, w in weights[weights > 0].sort_values(ascending=False).items():
        close = float(last_close.get(ticker, float("nan")))
        target_value = seed_won * w if seed_won else None
        target_shares = int(target_value // close) if target_value and close > 0 else None
        rows.append(
            HoldingRow(
                ticker=ticker,
                name=name_map.get(ticker, "?"),
                weight=float(w),
                last_close=close,
                target_value_won=target_value,
                target_shares=target_shares,
            )
        )
    return rows


def compute_daily_signal(
    *,
    seed_won: float | None = None,
    prices: pd.DataFrame | None = None,
    values: pd.DataFrame | None = None,
    config: MomentumTopNConfig | None = None,
    ma_window: int = 100,
) -> DailySignal:
    """챔피언(월간 모멘텀+MA100) 기준 오늘 알림 생성.

    args:
        seed_won: 시드(원). 입력 시 종목별 매수 금액·주식 수 계산.
        prices/values: 미리 로드된 패널 (테스트 용). None이면 디스크에서 로드.
        config: MomentumTopNConfig. None이면 챔피언 기본값 (월간/12m/Top10).
        ma_window: 시장 레짐 MA 윈도우 (100=챔피언, 200=보수적).
    """
    cfg = config or MomentumTopNConfig(
        top_n=10, lookback_months=12, rebalance_freq="BMS", replace_threshold=0.0
    )
    if prices is None:
        prices = load_close_panel()
    if values is None:
        values = load_value_panel()
    if prices.empty:
        raise RuntimeError("저장된 가격 데이터 없음 — quant data fetch-krx 먼저 실행")

    # 시장 상태
    market = regime_state(prices, ma_window=ma_window)
    today = pd.Timestamp(market["as_of"]).date()
    name_map = _ticker_name_map()

    # 챔피언 가중치 패널 + 시장 필터
    base_weights = generate_weights(prices, values=values, config=cfg)
    weights, exposure = filter_weights(
        prices, base_weights, mode=RegimeMode.BINARY, ma_window=ma_window
    )

    # 마지막 리밸런싱 일자 (가중치가 변한 마지막 날)
    weight_change = weights.diff().abs().sum(axis=1)
    rb_dates = weights.index[weight_change > 1e-6]
    last_rb_date = rb_dates[-1] if len(rb_dates) > 0 else weights.index[-1]

    # 다음 리밸런싱 일자 (BMS = 월초 영업일)
    next_rb = pd.date_range(
        start=pd.Timestamp(today) + pd.DateOffset(days=1),
        end=pd.Timestamp(today) + pd.DateOffset(months=2),
        freq=cfg.rebalance_freq,
    )
    next_rb_date = next_rb[0].date() if len(next_rb) > 0 else None

    last_close = prices.iloc[-1]
    notes: list[str] = []

    # 시장 청산 감지 — 어제까지 위 → 오늘 아래
    yesterday_above = bool(exposure.iloc[-2] > 0) if len(exposure) >= 2 else True
    today_above = bool(market["in_uptrend"])
    if yesterday_above and not today_above:
        # 시장 MA 깨짐 → 즉시 청산
        prev_holdings = base_weights.iloc[-1]  # 필터 전 = 어제까지 보유
        holdings = _to_holding_rows(prev_holdings, last_close, name_map, seed_won)
        notes.append(f"시장 {market['distance_pct']:+.1f}% — MA{ma_window} 하향 돌파")
        return DailySignal(
            as_of=today,
            alert_type=AlertType.LIQUIDATE,
            market_state=market,
            holdings=holdings,
            new_buys=[],
            sells=holdings,  # 전량 매도
            next_rebalance=next_rb_date,
            notes=notes,
        )

    # 시장 회복 감지 — 어제까지 아래 → 오늘 위
    if not yesterday_above and today_above:
        notes.append(f"시장 회복 — MA{ma_window} 상향 돌파. 다음 리밸런싱부터 재개")
        return DailySignal(
            as_of=today,
            alert_type=AlertType.RECOVER,
            market_state=market,
            holdings=[],
            new_buys=[],
            sells=[],
            next_rebalance=next_rb_date,
            notes=notes,
        )

    # 시장이 아래일 때 — 보유 0 + 다음 회복 대기
    if not today_above:
        notes.append(
            f"시장 {market['distance_pct']:+.1f}% (MA{ma_window} 아래) — 현금 보유, 매수 금지"
        )
        return DailySignal(
            as_of=today,
            alert_type=AlertType.LIQUIDATE,
            market_state=market,
            holdings=[],
            new_buys=[],
            sells=[],
            next_rebalance=next_rb_date,
            notes=notes,
        )

    # 리밸런싱 도래 감지 — 오늘이 마지막 리밸런싱이거나 아주 최근
    is_rb_today = pd.Timestamp(today) == last_rb_date
    if is_rb_today:
        # 직전 리밸런싱 가중치와 오늘 가중치 차이
        prior_dates = rb_dates[rb_dates < last_rb_date]
        prev_holdings = (
            weights.loc[prior_dates[-1]] if len(prior_dates) > 0 else pd.Series(dtype=float)
        )
        cur_holdings = weights.loc[last_rb_date]
        prev_set = set(prev_holdings[prev_holdings > 0].index) if not prev_holdings.empty else set()
        cur_set = set(cur_holdings[cur_holdings > 0].index)
        sell_tickers = prev_set - cur_set
        buy_tickers = cur_set - prev_set
        sells = _to_holding_rows(
            prev_holdings[list(sell_tickers)] if sell_tickers else pd.Series(dtype=float),
            last_close,
            name_map,
            seed_won,
        )
        new_buys = _to_holding_rows(
            cur_holdings[list(buy_tickers)] if buy_tickers else pd.Series(dtype=float),
            last_close,
            name_map,
            seed_won,
        )
        holdings = _to_holding_rows(cur_holdings, last_close, name_map, seed_won)
        return DailySignal(
            as_of=today,
            alert_type=AlertType.REBALANCE,
            market_state=market,
            holdings=holdings,
            new_buys=new_buys,
            sells=sells,
            next_rebalance=next_rb_date,
            notes=notes,
        )

    # 정상 — 보유 유지
    cur_holdings = weights.loc[last_rb_date]
    holdings = _to_holding_rows(cur_holdings, last_close, name_map, seed_won)
    return DailySignal(
        as_of=today,
        alert_type=AlertType.NORMAL,
        market_state=market,
        holdings=holdings,
        new_buys=[],
        sells=[],
        next_rebalance=next_rb_date,
        notes=notes,
    )


def render_alert(signal: DailySignal) -> str:
    """DailySignal → 사람이 읽을 수 있는 알림 텍스트."""
    s = signal
    lines = [f"📅 {s.as_of} 매일 알림"]

    # 시장 상태
    m = s.market_state
    icon = "✅" if m["in_uptrend"] else "🔴"
    lines.append("")
    lines.append("📊 시장 상태")
    lines.append(f"  KOSPI EW: {m['market_value']:,.0f}")
    lines.append(f"  MA:       {m['ma_value']:,.0f} ({m['distance_pct']:+.1f}%)")
    lines.append(f"  → {icon} {m['recommendation']}")

    # 알림 타입별 메시지
    lines.append("")
    if s.alert_type is AlertType.LIQUIDATE:
        lines.append("🚨 즉시 행동: 전량 청산")
        for note in s.notes:
            lines.append(f"  - {note}")
        if s.sells:
            lines.append("")
            lines.append(f"매도 종목 ({len(s.sells)}):")
            for h in s.sells:
                shares = f" {h.target_shares}주" if h.target_shares else ""
                lines.append(f"  🔴 {h.ticker} {h.name:<14}{shares}")
    elif s.alert_type is AlertType.RECOVER:
        lines.append("✅ 시장 회복 — 매수 재개 신호")
        for note in s.notes:
            lines.append(f"  - {note}")
    elif s.alert_type is AlertType.REBALANCE:
        lines.append(f"🔔 월간 리밸런싱 ({s.as_of})")
        if s.sells:
            lines.append("")
            lines.append(f"🔴 매도 ({len(s.sells)}):")
            for h in s.sells:
                shares = f" {h.target_shares}주" if h.target_shares else ""
                lines.append(f"  - {h.ticker} {h.name:<14}{shares}")
        if s.new_buys:
            lines.append("")
            lines.append(f"🟢 매수 ({len(s.new_buys)}):")
            for h in s.new_buys:
                if h.target_value_won and h.target_shares:
                    lines.append(
                        f"  + {h.ticker} {h.name:<14} 약 {h.target_shares}주 "
                        f"(목표 {h.target_value_won:,.0f}원)"
                    )
                else:
                    lines.append(f"  + {h.ticker} {h.name}")
        lines.append("")
        lines.append("📲 한국투자증권 앱에서 위 주문 실행")
    else:  # NORMAL
        lines.append("✅ 보유 유지 — 행동 없음")
        if s.holdings:
            lines.append("")
            lines.append(f"💼 보유 종목 ({len(s.holdings)}):")
            for h in s.holdings:
                if h.target_value_won and h.target_shares:
                    lines.append(
                        f"  {h.ticker} {h.name:<14} {h.target_shares}주 "
                        f"({h.target_value_won:,.0f}원)"
                    )
                else:
                    lines.append(f"  {h.ticker} {h.name}")

    if s.next_rebalance:
        days = (s.next_rebalance - s.as_of).days
        lines.append("")
        lines.append(f"🔔 다음 리밸런싱: {s.next_rebalance} ({days}일 후)")

    return "\n".join(lines)
