"""매일 아침 알림 통합 — 모의투자/실거래 운용 핵심.

챔피언 (다중 레짐 #6):
  강세장:    모멘텀 Top-10 (12-1 모멘텀)
  강세 변동성: 모멘텀 60% + 저변동성 40%
  약세장:    저변동성 Top-10 (방어주 보유)
  회복기:    모멘텀 50% (점진적 진입)

5가지 시나리오 자동 판별:
1. 정상     : 보유 유지 (90% 이상 해당)
2. 리밸런싱 : 월간 매매일 도래 — 매도/매수 종목 명시
3. 청산     : 시장 MA 깨짐 — 모멘텀 → 저변동성 갈아타기 (B 정책)
4. 회복     : 시장 MA 회복 — 다음 리밸런싱부터 매수 재개
5. Kill     : 일일 -3% 또는 누적 -20% — 비상 청산 (가드레일 §10)

사용:
    quant signal --daily            # 챔피언 기준 알림
    quant signal --daily --seed 10000000  # 시드 입력 시 잔고/주식수 추정
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

import pandas as pd

from quant.backtest.regime import RegimeMode, filter_weights, regime_state
from quant.backtest.regime_strategy import RegimePolicy, combine_by_regime
from quant.common.logger import logger
from quant.data.price.fetch_krx import load_close_panel, load_value_panel
from quant.strategies.low_volatility import LowVolatilityConfig
from quant.strategies.low_volatility import generate_weights as lv_w
from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights

# 챔피언 정책 #6 — 강세에 모멘텀, 약세엔 저변동성, 변동성 큰 강세엔 60/40
CHAMPION_POLICY_B = RegimePolicy(
    bull_strong={"momentum": 1.0},
    bull_choppy={"momentum": 0.6, "low_vol": 0.4},
    bear={"low_vol": 1.0},  # 약세장에 저변동성 100% 보유 (B 정책)
    recovery={"momentum": 0.5},  # 회복기 50% 진입 + 50% 현금
)


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
    ret_1d: float | None = None  # 어제 종가 대비 오늘 수익률 (양/음)
    ret_5d: float | None = None  # 5일 전 대비
    ret_1m: float | None = None  # 1개월 전 대비
    # KIS 실잔고 매칭 시 채워짐 (없으면 None)
    actual_shares: int | None = None
    avg_price: float | None = None  # 매입 평균가
    eval_amount: float | None = None  # 평가금액
    profit: float | None = None  # 평가손익 (원)
    profit_pct: float | None = None  # 매입가 대비 손익률 (%)


@dataclass
class DailySignal:
    """매일 아침 알림 통합 결과."""

    as_of: date
    alert_type: AlertType
    market_state: dict  # regime_state() 출력
    holdings: list[HoldingRow]  # 목표 (전략이 보유해야 할) 종목
    new_buys: list[HoldingRow]  # 리밸런싱 시 신규 매수
    sells: list[HoldingRow]  # 리밸런싱 시 매도
    next_rebalance: date | None  # 다음 리밸런싱일
    notes: list[str]  # 추가 알림 (Kill switch 사유 등)
    # KIS 실잔고 종합 (조회 실패 시 None)
    balance_total_eval: float | None = None  # 평가합계
    balance_total_cost: float | None = None  # 매입합계
    balance_total_profit: float | None = None  # 평가손익 합계
    balance_total_profit_pct: float | None = None  # 누적 평가손익률 (매입 대비)
    balance_change_1d_won: float | None = None  # 1일 변동액 (어제 평가 → 오늘 평가)
    balance_change_1d_pct: float | None = None  # 1일 변동률
    has_balance: bool = False  # 실잔고 조회 성공 여부


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


def _recent_return(prices: pd.DataFrame, ticker: str, days: int) -> float | None:
    """N영업일 전 대비 수익률. 데이터 부족·NaN 시 None."""
    if ticker not in prices.columns or len(prices) <= days:
        return None
    cur = prices[ticker].iloc[-1]
    past = prices[ticker].iloc[-1 - days]
    if pd.isna(cur) or pd.isna(past) or past <= 0:
        return None
    return float(cur / past - 1.0)


def _fetch_balance_safe() -> dict[str, object] | None:
    """KIS 잔고 조회 — 실패·미설정 시 None (graceful)."""
    try:
        from quant.live.client import get_balance

        holdings = get_balance()
        return {h.ticker: h for h in holdings}
    except Exception as e:
        logger.warning(f"KIS 잔고 조회 실패 (graceful): {e}")
        return None


def _to_holding_rows(
    weights: pd.Series,
    last_close: pd.Series,
    name_map: dict[str, str],
    seed_won: float | None,
    prices: pd.DataFrame | None = None,
    balance_map: dict | None = None,
) -> list[HoldingRow]:
    """가중치 → HoldingRow 리스트.

    balance_map (ticker → Holding) 주어지면 매입가/평가/손익 채움.
    """
    rows: list[HoldingRow] = []
    for ticker, w in weights[weights > 0].sort_values(ascending=False).items():
        close = float(last_close.get(ticker, float("nan")))
        target_value = seed_won * w if seed_won else None
        target_shares = int(target_value // close) if target_value and close > 0 else None
        ret_1d = _recent_return(prices, ticker, 1) if prices is not None else None
        ret_5d = _recent_return(prices, ticker, 5) if prices is not None else None
        ret_1m = _recent_return(prices, ticker, 21) if prices is not None else None
        actual_shares = None
        avg_price = None
        eval_amount = None
        profit = None
        profit_pct = None
        if balance_map and ticker in balance_map:
            h = balance_map[ticker]
            actual_shares = h.quantity
            avg_price = h.avg_price
            eval_amount = h.eval_amount
            profit = h.profit
            profit_pct = h.profit_pct
        rows.append(
            HoldingRow(
                ticker=ticker,
                name=name_map.get(ticker, "?"),
                weight=float(w),
                last_close=close,
                target_value_won=target_value,
                target_shares=target_shares,
                ret_1d=ret_1d,
                ret_5d=ret_5d,
                ret_1m=ret_1m,
                actual_shares=actual_shares,
                avg_price=avg_price,
                eval_amount=eval_amount,
                profit=profit,
                profit_pct=profit_pct,
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
    multi_regime: bool = False,
    fetch_balance: bool = True,
    balance_map: dict | None = None,
) -> DailySignal:
    """챔피언 기준 오늘 알림 생성.

    args:
        seed_won: 시드(원). 입력 시 종목별 매수 금액·주식 수 계산.
        prices/values: 미리 로드된 패널 (테스트 용). None이면 디스크에서 로드.
        config: MomentumTopNConfig. None이면 챔피언 기본값.
        ma_window: 시장 레짐 MA 윈도우 (100=챔피언, 200=보수적).
        multi_regime: True면 정책 #6 (BEAR=lv 100%) 사용. False면 단일 모멘텀+MA 청산.
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

    # KIS 실잔고 (있으면 매입가·평가·손익 알림에 포함)
    if balance_map is None and fetch_balance:
        balance_map = _fetch_balance_safe()

    # 시장 상태
    market = regime_state(prices, ma_window=ma_window)
    proxy = prices.mean(axis=1)
    if len(proxy) >= 2:
        market["change_1d"] = float(proxy.iloc[-1] / proxy.iloc[-2] - 1.0)
    if len(proxy) >= 6:
        market["change_5d"] = float(proxy.iloc[-1] / proxy.iloc[-6] - 1.0)
    today = pd.Timestamp(market["as_of"]).date()
    name_map = _ticker_name_map()

    # 챔피언 가중치 패널
    if multi_regime:
        # B 정책 — 약세장에 저변동성 보유
        mom_panel = generate_weights(prices, values=values, config=cfg)
        lv_panel = lv_w(
            prices,
            values=values,
            config=LowVolatilityConfig(
                top_n=10,
                vol_window_days=60,
                rebalance_freq=cfg.rebalance_freq,
            ),
        )
        weights = combine_by_regime(
            prices,
            {"momentum": mom_panel, "low_vol": lv_panel},
            policy=CHAMPION_POLICY_B,
        )
        # exposure은 단순 보유 여부 (binary): row sum > 0이면 1
        exposure = (weights.sum(axis=1) > 1e-6).astype(float)
        base_weights = weights  # 청산 메시지 시 보유 종목 = 현재
    else:
        # 단일 챔피언 (이전 default)
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

    # 잔고 합계 (실잔고가 있으면 채움, 없으면 None)
    bal_eval = bal_cost = bal_profit = bal_profit_pct = None
    bal_change_1d_won = bal_change_1d_pct = None
    has_balance = balance_map is not None and len(balance_map) > 0
    if has_balance:
        # 일관성 위해 모든 평가는 prices.iloc[-1] 종가 기반 (KIS current_price 대신)
        today_prices = prices.iloc[-1]
        bal_eval = sum(
            h.quantity * float(today_prices.get(h.ticker, h.current_price))
            for h in balance_map.values()
        )
        bal_cost = sum(h.avg_price * h.quantity for h in balance_map.values())
        bal_profit = bal_eval - bal_cost if bal_cost else None
        bal_profit_pct = (bal_profit / bal_cost * 100.0) if bal_cost else None
        # 1일 변동 — 직전 영업일 평가 vs 오늘 평가
        if len(prices) >= 2:
            yesterday_prices = prices.iloc[-2]
            yesterday_eval = sum(
                h.quantity * float(yesterday_prices.get(h.ticker, h.avg_price))
                for h in balance_map.values()
            )
            if yesterday_eval > 0:
                bal_change_1d_won = bal_eval - yesterday_eval
                bal_change_1d_pct = bal_change_1d_won / yesterday_eval * 100.0

    def _make_signal(**kw) -> DailySignal:
        """공통 필드 (잔고 총계 등) 채워서 DailySignal 생성."""
        return DailySignal(
            as_of=today,
            market_state=market,
            next_rebalance=next_rb_date,
            balance_total_eval=bal_eval,
            balance_total_cost=bal_cost,
            balance_total_profit=bal_profit,
            balance_total_profit_pct=bal_profit_pct,
            balance_change_1d_won=bal_change_1d_won,
            balance_change_1d_pct=bal_change_1d_pct,
            has_balance=has_balance,
            **kw,
        )

    # 시장 청산 감지 — multi_regime=false (단일 챔피언) 모드에서만
    # multi_regime=true면 BEAR 진입은 weights 변화로 REBALANCE로 자동 처리됨
    yesterday_above = bool(exposure.iloc[-2] > 0) if len(exposure) >= 2 else True
    today_above = bool(market["in_uptrend"])

    if not multi_regime:
        if yesterday_above and not today_above:
            # 시장 MA 깨짐 → 즉시 청산 (단일 챔피언 모드)
            prev_holdings = base_weights.iloc[-1]  # 필터 전 = 어제까지 보유
            holdings = _to_holding_rows(
                prev_holdings, last_close, name_map, seed_won, prices, balance_map
            )
            notes.append(f"시장 {market['distance_pct']:+.1f}% — MA{ma_window} 하향 돌파")
            return _make_signal(
                alert_type=AlertType.LIQUIDATE,
                holdings=holdings,
                new_buys=[],
                sells=holdings,  # 전량 매도
                notes=notes,
            )

        # 시장 회복 감지 — 어제까지 아래 → 오늘 위
        if not yesterday_above and today_above:
            notes.append(f"시장 회복 — MA{ma_window} 상향 돌파. 다음 리밸런싱부터 재개")
            return _make_signal(
                alert_type=AlertType.RECOVER,
                holdings=[],
                new_buys=[],
                sells=[],
                notes=notes,
            )

        # 시장이 아래일 때 — 보유 0 + 다음 회복 대기
        if not today_above:
            notes.append(
                f"시장 {market['distance_pct']:+.1f}% (MA{ma_window} 아래) — 현금 보유, 매수 금지"
            )
            return _make_signal(
                alert_type=AlertType.LIQUIDATE,
                holdings=[],
                new_buys=[],
                sells=[],
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
        # 매도 후보는 실잔고 매핑 함께 (현재 보유 가격/손익 알림용)
        sells = _to_holding_rows(
            prev_holdings[list(sell_tickers)] if sell_tickers else pd.Series(dtype=float),
            last_close,
            name_map,
            seed_won,
            prices,
            balance_map,
        )
        new_buys = _to_holding_rows(
            cur_holdings[list(buy_tickers)] if buy_tickers else pd.Series(dtype=float),
            last_close,
            name_map,
            seed_won,
            prices,
        )
        holdings = _to_holding_rows(
            cur_holdings, last_close, name_map, seed_won, prices, balance_map
        )
        return _make_signal(
            alert_type=AlertType.REBALANCE,
            holdings=holdings,
            new_buys=new_buys,
            sells=sells,
            notes=notes,
        )

    # 정상 — 보유 유지
    cur_holdings = weights.loc[last_rb_date]
    holdings = _to_holding_rows(cur_holdings, last_close, name_map, seed_won, prices, balance_map)
    return _make_signal(
        alert_type=AlertType.NORMAL,
        holdings=holdings,
        new_buys=[],
        sells=[],
        notes=notes,
    )


def _fmt_pct(v: float | None, *, plus: bool = True) -> str:
    """수익률 포맷 — '-' 처리 + 색 이모지."""
    if v is None:
        return "─"
    sign = "+" if (plus and v >= 0) else ""
    return f"{sign}{v * 100:.1f}%"


def _ret_icon(v: float | None) -> str:
    if v is None:
        return "  "
    if v > 0.005:
        return "🔵"
    if v < -0.005:
        return "🔴"
    return "⚪"


def _fmt_won(v: float | None) -> str:
    """원 → 만원 단위 (소수점 1자리). 한국 금융 관행. None 처리."""
    if v is None:
        return "─"
    return f"{v / 10000:,.1f}만"


def _render_balance_section(signal: DailySignal) -> list[str]:
    """KIS 실잔고 풀텍스트 섹션 (Format B)."""
    s = signal
    if not s.has_balance:
        return []

    lines: list[str] = [""]
    n = sum(1 for h in s.holdings if h.actual_shares)

    # 핵심 손익 1줄 — 1일 변동 + 누적
    chg_1d = ""
    if s.balance_change_1d_pct is not None:
        sign1d = "+" if s.balance_change_1d_won >= 0 else "−"
        chg_1d = (
            f"오늘 {s.balance_change_1d_pct:+.2f}% "
            f"({sign1d}{_fmt_won(abs(s.balance_change_1d_won))}) / "
        )
    cum = ""
    if s.balance_total_profit_pct is not None:
        sign_c = "+" if s.balance_total_profit >= 0 else "−"
        cum = (
            f"누적 {s.balance_total_profit_pct:+.2f}% "
            f"({sign_c}{_fmt_won(abs(s.balance_total_profit))})"
        )
    lines.append(f"💼 {chg_1d}{cum}")
    lines.append(
        f"   잔고 {n}종목 · 평가 {_fmt_won(s.balance_total_eval)} (매입 {_fmt_won(s.balance_total_cost)})"
    )

    # 실보유 종목만 (actual_shares > 0)
    actual = [h for h in s.holdings if h.actual_shares and h.actual_shares > 0]
    for h in actual:
        icon = _ret_icon(h.ret_1d)
        close = int(h.last_close) if h.last_close else 0
        avg = int(h.avg_price) if h.avg_price else 0
        pp = h.profit_pct if h.profit_pct is not None else 0.0
        eval_k = _fmt_won(h.eval_amount)
        lines.append(
            f"{icon} {h.name} {h.actual_shares}주 @{avg:,}→{close:,} " f"({pp:+.1f}%) {eval_k}"
        )

    # 시그널엔 있는데 아직 안 산 종목 (목표지만 미보유)
    not_bought = [h for h in s.holdings if not h.actual_shares]
    if not_bought:
        lines.append("")
        lines.append(f"📋 시그널 권고 + 미보유 {len(not_bought)}종목:")
        for h in not_bought:
            close_str = f"@{int(h.last_close):,}" if h.last_close else ""
            shares = h.target_shares or 0
            if shares > 0:
                lines.append(f"   {h.name} 목표 {shares}주 {close_str}")
            else:
                price_man = (h.last_close or 0) / 10000
                lines.append(f"   {h.name} 1주 {price_man:,.0f}만 (시드 부족)")

    # TOP/BOT 손익률 (실잔고 기준)
    valid = [h for h in actual if h.profit_pct is not None]
    if len(valid) >= 2:
        best = max(valid, key=lambda x: x.profit_pct)
        worst = min(valid, key=lambda x: x.profit_pct)
        lines.append("")
        lines.append(f"🏆 최고: {best.name} {best.profit_pct:+.1f}%")
        lines.append(f"⚠️ 최저: {worst.name} {worst.profit_pct:+.1f}%")

    return lines


def _render_target_section(signal: DailySignal) -> list[str]:
    """KIS 잔고 없을 때 — 목표 포트폴리오 표시 (예전 형식 유지)."""
    s = signal
    if not s.holdings:
        return []

    lines: list[str] = [""]
    lines.append(f"📦 목표 포트폴리오 {len(s.holdings)}종목 (시뮬 — 실잔고 아님)")
    insufficient: list[HoldingRow] = []
    for h in s.holdings:
        ret_5d = _fmt_pct(h.ret_5d)
        ret_1d = _fmt_pct(h.ret_1d)
        if h.target_shares and h.target_shares > 0 and h.last_close:
            val_man = (h.target_shares * h.last_close) / 10000
            lines.append(
                f"{_ret_icon(h.ret_1d)} {h.name} "
                f"{h.target_shares}주 {val_man:,.0f}만 ({ret_1d}/5d {ret_5d})"
            )
        else:
            insufficient.append(h)
    if insufficient:
        lines.append("")
        lines.append(f"⚠️ 시드 부족 {len(insufficient)}종목 (1주 가격 > 종목당 분배금):")
        for h in insufficient:
            price_man = (h.last_close or 0) / 10000
            lines.append(f"   {h.name} 1주 {price_man:,.0f}만")
    return lines


def render_alert(signal: DailySignal) -> str:
    """텔레그램용 알림 (Format B — 실잔고 + 손익 중심).

    실잔고 있으면: 잔고 풀텍스트 + 종목별 손익 + TOP/BOT.
    실잔고 없으면: 목표 포트폴리오 (시뮬 명시).
    """
    s = signal
    m = s.market_state

    # ----- 헤더 — 시그널 타입을 명시적으로 (NORMAL/REBALANCE/LIQUIDATE/...) -----
    title_map = {
        AlertType.NORMAL: "✅ NORMAL — 보유 유지",
        AlertType.REBALANCE: "🔔 REBALANCE — 리밸런싱 매매일",
        AlertType.LIQUIDATE: "🚨 LIQUIDATE — 청산",
        AlertType.RECOVER: "🟢 RECOVER — 시장 회복",
        AlertType.KILL_SWITCH: "🚨 KILL SWITCH — 비상",
    }
    title = title_map.get(s.alert_type, "ℹ️")
    lines = [f"🌅 {s.as_of} {title}"]

    # ----- 시장 한 줄 -----
    pos = "MA위" if m["in_uptrend"] else "MA아래"
    safety = "안전" if m["in_uptrend"] else "위험"
    chg_1d = m.get("change_1d")
    chg_5d = m.get("change_5d")
    chg_text = ""
    if chg_1d is not None:
        chg_text = f" 어제 {_fmt_pct(chg_1d)}"
    if chg_5d is not None:
        chg_text += f" / 5d {_fmt_pct(chg_5d)}"
    lines.append(f"📈 KOSPI {pos} {m['distance_pct']:+.1f}% ({safety}){chg_text}")

    # ----- 본문 (잔고 있으면 잔고 위주, 없으면 목표) -----
    if s.has_balance:
        lines.extend(_render_balance_section(s))
    else:
        lines.extend(_render_target_section(s))

    # ----- 행동 지시 -----
    if s.alert_type is AlertType.LIQUIDATE:
        lines.append("")
        lines.append("🚨 즉시 전량 청산 — 시장 약세 진입")
        for note in s.notes:
            lines.append(f"  · {note}")
        if s.sells:
            lines.append(f"\n매도 {len(s.sells)}종목:")
            for h in s.sells:
                shares = h.actual_shares or h.target_shares or 0
                lines.append(f"  🔴 {h.ticker} {h.name} {shares}주")

    elif s.alert_type is AlertType.RECOVER:
        lines.append("")
        lines.append("🟢 시장 회복 — 다음 리밸런싱일에 매수 재개")
        for note in s.notes:
            lines.append(f"  · {note}")

    elif s.alert_type is AlertType.REBALANCE:
        lines.append("")
        lines.append(f"🎯 자동 매매 09:30 ({len(s.sells)}↔{len(s.new_buys)})")
        if s.sells:
            lines.append("\n🔴 매도:")
            for h in s.sells:
                shares = h.actual_shares or h.target_shares or 0
                pp_text = f" ({h.profit_pct:+.1f}%)" if h.profit_pct is not None else ""
                lines.append(f"  · {h.ticker} {h.name} {shares}주{pp_text}")
        if s.new_buys:
            lines.append("\n🟢 매수:")
            for h in s.new_buys:
                if h.target_value_won and h.target_shares:
                    ret_1m = _fmt_pct(h.ret_1m)
                    lines.append(
                        f"  · {h.ticker} {h.name} {h.target_shares}주 "
                        f"({h.target_value_won // 10000:,.0f}만원, 1m {ret_1m})"
                    )
                else:
                    lines.append(f"  · {h.ticker} {h.name}")

    else:  # NORMAL
        lines.append("")
        lines.append("🎯 행동: 보유 유지")

    # ----- 푸터 -----
    if s.next_rebalance:
        days = (s.next_rebalance - s.as_of).days
        lines.append(f"🔔 다음 매매: {s.next_rebalance} ({days}일 후)")

    return "\n".join(lines)
