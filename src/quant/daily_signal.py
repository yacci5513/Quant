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

from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum

import pandas as pd

from quant.backtest.regime import regime_state
from quant.backtest.regime_strategy import RegimePolicy, combine_by_regime
from quant.common.logger import logger
from quant.data.price.fetch_krx import load_close_panel, load_value_panel
from quant.strategies.low_volatility import LowVolatilityConfig
from quant.strategies.low_volatility import generate_weights as lv_w
from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights

# 단일 종목 최대 노출 (시드 대비 비율) — 1주 가격이 이걸 초과하면 매수 차단
_MAX_POSITION_PCT = 0.20
# 시드 부족 종목을 자동 대체하기 위한 spare 종목 수 (Top-N+spare 후보 풀)
_SPARE_TOP_N = 5


def _select_buyable_topn(
    weights_row: pd.Series,
    last_close: pd.Series,
    seed_won: float | None,
    target_n: int,
) -> pd.Series:
    """가중치 row → 매수 가능한 상위 target_n 종목만 균등 가중치로 재구성.

    1주 가격이 시드 × MAX_POSITION_PCT 초과 (시드 부족) 종목 자동 제외.
    남은 종목 중 원래 가중치 큰 순 target_n까지 선정 → 각 1/target_n 균등.
    """
    if seed_won is None or seed_won <= 0:
        return weights_row
    candidates = weights_row[weights_row > 0].sort_values(ascending=False)
    buyable: list[str] = []
    threshold = seed_won * _MAX_POSITION_PCT
    for ticker in candidates.index:
        close = float(last_close.get(ticker, float("nan")))
        if close > 0 and close <= threshold:
            buyable.append(ticker)
        if len(buyable) >= target_n:
            break
    if not buyable:
        return weights_row * 0.0  # 매수 가능 종목 0 → 전부 비중 0
    w = 1.0 / len(buyable)
    result = pd.Series(0.0, index=weights_row.index)
    for ticker in buyable:
        result[ticker] = w
    return result


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
    # KIS 실잔고 전체 (시그널 종목과 무관, 잔고 알림에 표시됨)
    actual_holdings: list[HoldingRow] = field(default_factory=list)
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
        # 종목 1주 가격이 시드의 _MAX_POSITION_PCT 넘으면 매수 차단 (단일 종목 집중 방지)
        if seed_won and close > 0 and close > seed_won * _MAX_POSITION_PCT:
            # 1주 가격이 시드의 20% 초과 → 매수 차단
            target_shares = 0
        elif target_value and close > 0:
            target_shares = int(target_value // close)
        else:
            target_shares = None
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
        # 단일 챔피언 — Top-(N+spare)로 확장해서 시드 부족 종목 자동 대체 (cur_holdings 단계)
        # MA100 시장 청산 가드 제거: 강세장 외삽 상태에선 MA100이 너무 멀어 의미 없음.
        # 종목별 손절·익절·트레일링 + 포트 Kill switch가 위험 관리 (execute.py)
        cfg_spare = replace(cfg, top_n=cfg.top_n + _SPARE_TOP_N)
        base_weights = generate_weights(prices, values=values, config=cfg_spare)
        weights = base_weights
        exposure = (weights.sum(axis=1) > 1e-6).astype(float)

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

    # 잔고 합계 + 실잔고 종목 리스트
    bal_eval = bal_cost = bal_profit = bal_profit_pct = None
    bal_change_1d_won = bal_change_1d_pct = None
    actual_holdings_rows: list[HoldingRow] = []
    has_balance = balance_map is not None and len(balance_map) > 0
    if has_balance:
        # KIS 응답 기반 1일 변동 — 종목별 quote.change (전일 대비 원 차이) × 수량 합계
        # KIS API 호출 추가됨 (잔고 종목 ~10개라 부담 적음)
        per_ticker_change_won: dict[str, float] = {}
        try:
            from quant.live.client import get_quote

            for ticker in balance_map:
                try:
                    q = get_quote(ticker)
                    per_ticker_change_won[ticker] = q.change  # 전일 대비 원
                except Exception:
                    per_ticker_change_won[ticker] = 0.0
        except Exception as e:
            logger.warning(f"KIS 1d quote 조회 실패 (graceful): {e}")

        # 실잔고 모든 종목을 HoldingRow로 변환 (KIS 데이터 일관)
        for ticker, h in balance_map.items():
            change_won = per_ticker_change_won.get(ticker, 0.0)
            # 종목별 1일 변동률 — (current - prdy_close) / prdy_close
            prdy_close = h.current_price - change_won if change_won else None
            ret_1d_kis = (change_won / prdy_close) if prdy_close and prdy_close > 0 else None
            actual_holdings_rows.append(
                HoldingRow(
                    ticker=ticker,
                    name=name_map.get(ticker, h.name or "?"),
                    weight=0.0,
                    last_close=h.current_price,  # KIS 실시간 (손익률과 일관)
                    ret_1d=ret_1d_kis,
                    ret_5d=_recent_return(prices, ticker, 5),
                    ret_1m=_recent_return(prices, ticker, 21),
                    actual_shares=h.quantity,
                    avg_price=h.avg_price,
                    eval_amount=h.eval_amount,
                    profit=h.profit,
                    profit_pct=h.profit_pct,
                )
            )
        # 잔고 합계 — KIS 응답 직접 사용 (current_price × quantity 또는 evlu_amt)
        bal_eval = sum(h.eval_amount for h in balance_map.values())
        bal_cost = sum(h.avg_price * h.quantity for h in balance_map.values())
        bal_profit = bal_eval - bal_cost if bal_cost else None
        bal_profit_pct = (bal_profit / bal_cost * 100.0) if bal_cost else None
        # 1일 변동 — 종목별 change_won × quantity 합계
        bal_change_1d_won = sum(
            per_ticker_change_won.get(h.ticker, 0.0) * h.quantity for h in balance_map.values()
        )
        if bal_eval > 0:
            # (오늘 평가 - 어제 평가) / 어제 평가 = change / (eval - change)
            yesterday_eval = bal_eval - bal_change_1d_won
            bal_change_1d_pct = (
                (bal_change_1d_won / yesterday_eval * 100.0) if yesterday_eval > 0 else None
            )

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
            actual_holdings=actual_holdings_rows,
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

    # 리밸런싱 판정 — 실잔고 vs 시그널 목표 차이 (가중치 변경 자체 X)
    # multi_regime=true에서 매일 weights가 미세하게 바뀌는 노이즈를 무시하기 위함
    cur_holdings_raw = weights.loc[last_rb_date]
    # 시드 부족 종목 자동 대체 — Top-(N+spare) 중 매수 가능 상위 N 선정 + 균등 가중
    cur_holdings = _select_buyable_topn(cur_holdings_raw, last_close, seed_won, cfg.top_n)
    # 시그널 종목 중 매수 가능한 것만 (target_shares > 0)
    target_rows = _to_holding_rows(
        cur_holdings, last_close, name_map, seed_won, prices, balance_map
    )
    target_buyable_set = {h.ticker for h in target_rows if h.target_shares and h.target_shares > 0}
    actual_set = {h.ticker for h in actual_holdings_rows}

    sell_tickers = actual_set - target_buyable_set
    buy_tickers = target_buyable_set - actual_set
    needs_action = bool(sell_tickers) or bool(buy_tickers)

    if needs_action:
        # 매도: 실잔고에서 시그널이 빠진 종목 (KIS 매입가·평가 포함)
        sells = [h for h in actual_holdings_rows if h.ticker in sell_tickers]
        # 매수: 시그널에 새로 들어온 종목
        new_buys = [h for h in target_rows if h.ticker in buy_tickers]
        return _make_signal(
            alert_type=AlertType.REBALANCE,
            holdings=target_rows,
            new_buys=new_buys,
            sells=sells,
            notes=notes,
        )

    # 정상 — 실잔고가 시그널과 일치
    return _make_signal(
        alert_type=AlertType.NORMAL,
        holdings=target_rows,
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
    n = len(s.actual_holdings)

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

    # 실잔고 모든 종목 (매도 예정 종목도 포함) — KIS balance 전부 표시
    sell_set = {h.ticker for h in s.sells}
    for h in s.actual_holdings:
        icon = _ret_icon(h.ret_1d)
        close = int(h.last_close) if h.last_close else 0
        avg = int(h.avg_price) if h.avg_price else 0
        pp = h.profit_pct if h.profit_pct is not None else 0.0
        eval_k = _fmt_won(h.eval_amount)
        mark = " 🔴매도예정" if h.ticker in sell_set else ""
        lines.append(
            f"{icon} {h.name} {h.actual_shares}주 @{avg:,}→{close:,} "
            f"({pp:+.1f}%) {eval_k}{mark}"
        )

    # 시그널 권고 중 아직 미보유 종목 (매수 예정)
    actual_tickers = {h.ticker for h in s.actual_holdings}
    not_bought = [h for h in s.holdings if h.ticker not in actual_tickers]
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
    valid = [h for h in s.actual_holdings if h.profit_pct is not None]
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


# ============================================================================
# Dual Portfolio (챔피언 + 어그레시브 D) — Compact 알림
# ============================================================================


@dataclass
class DualSignalConfig:
    """이중 포트폴리오 설정 — 챔피언(안정) + 어그레시브(D 시나리오)."""

    champion_pct: float = 0.70
    aggressive_pct: float = 0.30
    # 챔피언: 월간 12m Top-10 + MA100 — walk-forward §10 통과 (Sharpe 1.34, MDD -23.5%)
    champion_top_n: int = 10
    champion_lookback: int = 12
    champion_skip: int = 1
    champion_freq: str = "BMS"
    # 어그레시브: 주간 1m Top-5 + MA100 — walk-forward Sharpe 2.69, MDD -32% (§10 위반, 시드 일부만)
    aggressive_top_n: int = 5
    aggressive_lookback: int = 1
    aggressive_skip: int = 0
    aggressive_freq: str = "W-FRI"
    ma_window: int = 100


@dataclass
class DualSignal:
    """챔피언 + 어그레시브 통합 시그널.

    두 전략 시그널 합집합으로 통합 매도/매수 액션 산출.
    KIS 잔고는 1계좌이므로 어느 종목이 어느 전략인지 추적 불가 →
    "시그널 권고 합집합 - 잔고 = 매수 / 잔고 - 합집합 = 매도" 단순화.
    """

    as_of: date
    market_state: dict
    champion: DailySignal
    aggressive: DailySignal
    dual_config: DualSignalConfig
    combined_sells: list[HoldingRow] = field(default_factory=list)
    combined_buys: list[HoldingRow] = field(default_factory=list)
    balance_total_eval: float | None = None
    balance_total_cost: float | None = None
    balance_total_profit: float | None = None
    balance_total_profit_pct: float | None = None
    balance_change_1d_won: float | None = None
    balance_change_1d_pct: float | None = None
    has_balance: bool = False


def compute_dual_signal(
    *,
    seed_won: float | None = None,
    dual_config: DualSignalConfig | None = None,
    prices: pd.DataFrame | None = None,
    values: pd.DataFrame | None = None,
    fetch_balance: bool = True,
    balance_map: dict | None = None,
) -> DualSignal:
    """챔피언 + 어그레시브 두 시그널 동시 계산.

    시드는 champion_pct / aggressive_pct로 분할. 각 전략은 자기 시드 안에서
    매수 가능 종목을 선정. KIS 잔고는 통합 — 어느 종목이 어느 전략인지는
    시그널 매핑으로 추적 (알림에 표시).

    balance_map: 외부에서 fetch한 잔고를 주입하면 중복 호출 회피 (execute.py 등).
    """
    cfg = dual_config or DualSignalConfig()
    if prices is None:
        prices = load_close_panel()
    if values is None:
        values = load_value_panel()
    if prices.empty:
        raise RuntimeError("저장된 가격 데이터 없음 — quant data fetch-krx 먼저 실행")

    if balance_map is None and fetch_balance:
        balance_map = _fetch_balance_safe()

    champion_cfg = MomentumTopNConfig(
        top_n=cfg.champion_top_n,
        lookback_months=cfg.champion_lookback,
        skip_months=cfg.champion_skip,
        rebalance_freq=cfg.champion_freq,
    )
    champ_seed = seed_won * cfg.champion_pct if seed_won else None
    champion = compute_daily_signal(
        seed_won=champ_seed,
        prices=prices,
        values=values,
        config=champion_cfg,
        ma_window=cfg.ma_window,
        multi_regime=False,
        fetch_balance=False,
        balance_map=balance_map,
    )

    aggressive_cfg = MomentumTopNConfig(
        top_n=cfg.aggressive_top_n,
        lookback_months=cfg.aggressive_lookback,
        skip_months=cfg.aggressive_skip,
        rebalance_freq=cfg.aggressive_freq,
    )
    agg_seed = seed_won * cfg.aggressive_pct if seed_won else None
    aggressive = compute_daily_signal(
        seed_won=agg_seed,
        prices=prices,
        values=values,
        config=aggressive_cfg,
        ma_window=cfg.ma_window,
        multi_regime=False,
        fetch_balance=False,
        balance_map=balance_map,
    )

    # ----- 통합 매도/매수 액션 (매매일 룰 적용) -----
    # 챔피언은 매월 1일 (BMS), 어그는 매주 금 (W-FRI)만 매매.
    # 비매매일에는 시그널 매도 비활성 — 보유 유지 (손절·익절·Kill switch만 별도 적용).
    # 매수는 매매일에만 신규 진입 — 다음 매매일까지 시드 미달 인정.
    today_ts = prices.index[-1]
    champ_panel_full = generate_weights(prices, values=values, config=champion_cfg)
    agg_panel_full = generate_weights(prices, values=values, config=aggressive_cfg)

    def _is_rebalance_today(panel: pd.DataFrame) -> bool:
        if len(panel) < 2 or today_ts not in panel.index:
            return False
        prev_idx = panel.index[panel.index < today_ts]
        if len(prev_idx) == 0:
            return False
        diff = (panel.loc[today_ts] - panel.loc[prev_idx[-1]]).abs().sum()
        return float(diff) > 1e-6

    champ_is_rb = _is_rebalance_today(champ_panel_full)
    agg_is_rb = _is_rebalance_today(agg_panel_full)

    champ_targets = {h.ticker for h in champion.holdings if h.target_shares and h.target_shares > 0}
    agg_targets = {h.ticker for h in aggressive.holdings if h.target_shares and h.target_shares > 0}
    actual_set = {h.ticker for h in champion.actual_holdings}

    # 매도: 매매일에만 활성. 단 다른 전략이 권고하는 종목은 보호.
    sells_tickers: set[str] = set()
    if champ_is_rb:
        sells_tickers |= actual_set - champ_targets - agg_targets
    # 어그 매매일에만 정리하면 챔피언 시드 종목까지 침범 우려 → 어그 매매일 시그널 매도는 비활성
    # (어그 종목 누적 정리는 후속: state 기반 추적 필요)
    combined_sells = [h for h in champion.actual_holdings if h.ticker in sells_tickers]

    # 매수: 각 전략 매매일에만 활성. 시드 미달 자동 채움.
    buys_candidates: list[HoldingRow] = []
    if champ_is_rb:
        for h in champion.holdings:
            if h.ticker not in actual_set:
                buys_candidates.append(h)
    if agg_is_rb:
        for h in aggressive.holdings:
            if h.ticker not in actual_set:
                buys_candidates.append(h)

    # 중복 제거 + 보통주/우선주 동시 매수 금지
    held_after_sell = actual_set - {h.ticker for h in combined_sells}
    seen_roots: set[str] = {t[:5] for t in held_after_sell if len(t) >= 5}
    seen_tickers: set[str] = set()
    combined_buys: list[HoldingRow] = []
    for h in buys_candidates:
        if h.ticker in seen_tickers:
            continue
        root = h.ticker[:5] if len(h.ticker) >= 5 else h.ticker
        if root in seen_roots:
            continue
        combined_buys.append(h)
        seen_tickers.add(h.ticker)
        seen_roots.add(root)

    # 보통주/우선주 동시 매수 금지 — 종목코드 첫 5자리(prefix)가 같으면 같은 회사
    # 예) 006800 미래에셋증권 + 00680K 미래에셋증권2우B → 1개만 보유
    held_after_sell = actual_set - {h.ticker for h in combined_sells}
    seen_roots: set[str] = {t[:5] for t in held_after_sell if len(t) >= 5}
    deduped_buys: list[HoldingRow] = []
    for h in combined_buys:
        root = h.ticker[:5] if len(h.ticker) >= 5 else h.ticker
        if root in seen_roots:
            continue  # 같은 회사 종목 이미 보유 또는 다른 매수에 포함
        deduped_buys.append(h)
        seen_roots.add(root)
    combined_buys = deduped_buys

    return DualSignal(
        as_of=champion.as_of,
        market_state=champion.market_state,
        champion=champion,
        aggressive=aggressive,
        dual_config=cfg,
        combined_sells=combined_sells,
        combined_buys=combined_buys,
        balance_total_eval=champion.balance_total_eval,
        balance_total_cost=champion.balance_total_cost,
        balance_total_profit=champion.balance_total_profit,
        balance_total_profit_pct=champion.balance_total_profit_pct,
        balance_change_1d_won=champion.balance_change_1d_won,
        balance_change_1d_pct=champion.balance_change_1d_pct,
        has_balance=champion.has_balance,
    )


# ============================================================================
# Compact 렌더 — 이모지 최소, 섹션 구분선, 행동 상단, 포맷 일관
# ============================================================================

_SEP = "━━━━━━━━━━━━━━━━━━━━━"


def _hdr(title: str) -> str:
    return f"━━━ {title} ━━━"


def _fmt_won_compact(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v / 10000:,.1f}만"


def _fmt_signed_won(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(v) / 10000:,.0f}만"


def _holding_line(h: HoldingRow) -> str:
    """종목 1줄 — 이름 N주  +X.X% (+8만). 단일 포맷."""
    shares = h.actual_shares if h.actual_shares is not None else (h.target_shares or 0)
    name = h.name[:10]
    if h.profit_pct is not None:
        pct_str = f"{h.profit_pct:+5.1f}%"
        money_str = f" ({_fmt_signed_won(h.profit)})" if h.profit is not None else ""
    elif h.ret_1d is not None:
        pct_str = f"{h.ret_1d * 100:+5.1f}%"
        money_str = ""
    else:
        pct_str = "  —  "
        money_str = ""
    return f"  {name:<10} {shares:>4}주  {pct_str}{money_str}"


def _buy_line(h: HoldingRow) -> str:
    """매수 종목 1줄 — 이름 N주 (X만)."""
    name = h.name[:10]
    shares = h.target_shares or 0
    val = h.target_value_won
    val_str = f" ({val / 10000:,.0f}만)" if val else ""
    return f"  · 매수 {name:<10} {shares:>4}주{val_str}"


def _sell_line(h: HoldingRow) -> str:
    """매도 종목 1줄 — 이름 N주 (+X.X%)."""
    name = h.name[:10]
    shares = h.actual_shares or h.target_shares or 0
    pct = f" ({h.profit_pct:+.1f}%)" if h.profit_pct is not None else ""
    return f"  · 매도 {name:<10} {shares:>4}주{pct}"


def _render_strategy_section(
    name: str,
    pct: int,
    desc: str,
    sig: DailySignal,
    warn: str | None = None,
) -> list[str]:
    """전략 1개 섹션 — 시그널 권고 + 매칭 잔고만 표시 (매매 액션은 통합 섹션에서).

    종목 리스트:
      · 보유 중: 종목명 N주 +X.X% (+/-N만)
      · 신규 권고: 종목명 N주 (X만) 신규
      · 시드 부족: 종목명 1주 X만 (시드 부족)
    """
    out = ["", _hdr(f"{name} {pct}%")]
    desc_line = desc + (f"  ({warn})" if warn else "")
    out.append(desc_line)

    actual_map = {h.ticker: h for h in sig.actual_holdings}
    matched = [h for h in sig.holdings if h.ticker in actual_map]
    eval_sum = sum((actual_map[h.ticker].eval_amount or 0) for h in matched)
    cost_sum = sum(
        (actual_map[h.ticker].avg_price or 0) * (actual_map[h.ticker].actual_shares or 0)
        for h in matched
    )
    prof_pct = ((eval_sum - cost_sum) / cost_sum * 100) if cost_sum > 0 else None

    if matched:
        prof_str = f" ({prof_pct:+.1f}%)" if prof_pct is not None else ""
        out.append(
            f"평가 {_fmt_won_compact(eval_sum)}{prof_str} | "
            f"매칭 {len(matched)}/{len(sig.holdings)}종목"
        )
    else:
        out.append(f"시그널 권고 {len(sig.holdings)}종목 (현재 보유 0)")

    out.append("")
    for h in sig.holdings[:10]:
        if h.ticker in actual_map:
            out.append(_holding_line(actual_map[h.ticker]))
        else:
            name_str = h.name[:10]
            shares = h.target_shares or 0
            if shares > 0 and h.target_value_won:
                out.append(
                    f"  {name_str:<10} {shares:>4}주  ({h.target_value_won / 10000:,.0f}만) 신규"
                )
            elif h.last_close:
                price_man = h.last_close / 10000
                out.append(f"  {name_str:<10}    —  ({price_man:,.0f}만/주, 시드 부족)")
            else:
                out.append(f"  {name_str:<10}    —  신규")

    rb = sig.next_rebalance
    if rb:
        days = (rb - sig.as_of).days
        out.append("")
        out.append(f"다음 매매: {rb} ({days}일)")
    return out


def _render_top_actions(dual: DualSignal) -> list[str]:
    """최상단 — 오늘 통합 매매 요약 1~2줄."""
    out = [_hdr("오늘의 행동")]
    n_sell = len(dual.combined_sells)
    n_buy = len(dual.combined_buys)
    if n_sell or n_buy:
        out.append(f"통합 매매  매도 {n_sell} / 매수 {n_buy}  (09:30 자동)")
    else:
        out.append("보유 유지 (오늘 매매 없음)")

    next_parts = []
    if dual.champion.next_rebalance:
        days_c = (dual.champion.next_rebalance - dual.as_of).days
        next_parts.append(f"챔피언 {days_c}일")
    if dual.aggressive.next_rebalance:
        days_a = (dual.aggressive.next_rebalance - dual.as_of).days
        next_parts.append(f"어그 {days_a}일")
    if next_parts:
        out.append("다음 매매:  " + "  /  ".join(next_parts))
    return out


def _render_combined_actions(dual: DualSignal) -> list[str]:
    """통합 매매 액션 — 매도/매수 종목 상세 (실행 가능 형태)."""
    if not dual.combined_sells and not dual.combined_buys:
        return []
    out = [
        "",
        _hdr(f"매매 상세  매도 {len(dual.combined_sells)} / 매수 {len(dual.combined_buys)}"),
    ]
    if dual.combined_sells:
        out.append("매도:")
        for h in dual.combined_sells:
            out.append(_sell_line(h))
    if dual.combined_buys:
        out.append("매수:")
        for h in dual.combined_buys:
            out.append(_buy_line(h))
    return out


def _render_market_section(market: dict, ma_window: int) -> list[str]:
    pos = "위" if market["in_uptrend"] else "아래"
    safety = "안전" if market["in_uptrend"] else "위험"
    chg_1d = market.get("change_1d")
    chg_5d = market.get("change_5d")
    line2 = []
    if chg_1d is not None:
        line2.append(f"어제 {chg_1d * 100:+.1f}%")
    if chg_5d is not None:
        line2.append(f"5d {chg_5d * 100:+.1f}%")
    return [
        "",
        _hdr("시장"),
        f"KOSPI MA{ma_window} {pos} {market['distance_pct']:+.1f}% ({safety})",
        " / ".join(line2) if line2 else "",
    ]


def _render_balance_section_compact(dual: DualSignal) -> list[str]:
    if not dual.has_balance:
        return ["", _hdr("통합 잔고"), "KIS 잔고 미연동 (시뮬레이션 모드)"]
    eval_str = _fmt_won_compact(dual.balance_total_eval)
    cost_str = _fmt_won_compact(dual.balance_total_cost)
    today = ""
    if dual.balance_change_1d_pct is not None:
        today = f"오늘 {dual.balance_change_1d_pct:+.2f}%"
        if dual.balance_change_1d_won is not None:
            today += f" ({_fmt_signed_won(dual.balance_change_1d_won)})"
    cum = ""
    if dual.balance_total_profit_pct is not None:
        cum = f"누적 {dual.balance_total_profit_pct:+.2f}%"
        if dual.balance_total_profit is not None:
            cum += f" ({_fmt_signed_won(dual.balance_total_profit)})"
    parts = " / ".join(p for p in (today, cum) if p)
    return [
        "",
        _hdr("통합 잔고"),
        f"평가 {eval_str}원 (매입 {cost_str})",
        parts,
    ]


def _render_extremes_section(dual: DualSignal) -> list[str]:
    """전체 보유 종목 중 최고·최저 손익률 1줄씩."""
    all_actual = list(
        {
            h.ticker: h for h in (dual.champion.actual_holdings + dual.aggressive.actual_holdings)
        }.values()
    )
    valid = [h for h in all_actual if h.profit_pct is not None]
    if len(valid) < 2:
        return []
    best = max(valid, key=lambda x: x.profit_pct)
    worst = min(valid, key=lambda x: x.profit_pct)
    return [
        "",
        _hdr("손익 종목"),
        f"최고: {best.name} {best.profit_pct:+.1f}%",
        f"최저: {worst.name} {worst.profit_pct:+.1f}%",
    ]


@dataclass
class HourlyBalance:
    """장중 매시간 보유 손익 스냅샷."""

    as_of: pd.Timestamp
    holdings: list[HoldingRow]
    balance_total_eval: float | None
    balance_total_cost: float | None
    balance_total_profit: float | None
    balance_total_profit_pct: float | None
    balance_change_1d_won: float | None
    balance_change_1d_pct: float | None


def compute_hourly_balance() -> HourlyBalance:
    """KIS 잔고 + 종목별 1일 변동만 가볍게 수집. 시그널 계산 안 함 (빠름)."""
    from quant.live.client import get_balance, get_quote

    holdings_list = get_balance()
    balance_map = {h.ticker: h for h in holdings_list}
    name_map = _ticker_name_map()

    per_ticker_change_won: dict[str, float] = {}
    for h in holdings_list:
        try:
            q = get_quote(h.ticker)
            per_ticker_change_won[h.ticker] = q.change
        except Exception:
            per_ticker_change_won[h.ticker] = 0.0

    rows: list[HoldingRow] = []
    for ticker, h in balance_map.items():
        change_won = per_ticker_change_won.get(ticker, 0.0)
        prdy_close = h.current_price - change_won if change_won else None
        ret_1d = (change_won / prdy_close) if prdy_close and prdy_close > 0 else None
        rows.append(
            HoldingRow(
                ticker=ticker,
                name=name_map.get(ticker, h.name or "?"),
                weight=0.0,
                last_close=h.current_price,
                ret_1d=ret_1d,
                actual_shares=h.quantity,
                avg_price=h.avg_price,
                eval_amount=h.eval_amount,
                profit=h.profit,
                profit_pct=h.profit_pct,
            )
        )

    bal_eval = sum(h.eval_amount for h in balance_map.values()) if balance_map else None
    bal_cost = sum(h.avg_price * h.quantity for h in balance_map.values()) if balance_map else None
    bal_profit = (bal_eval - bal_cost) if (bal_eval is not None and bal_cost) else None
    bal_profit_pct = (
        (bal_profit / bal_cost * 100.0) if (bal_profit is not None and bal_cost) else None
    )
    bal_change_1d_won = (
        sum(per_ticker_change_won.get(h.ticker, 0.0) * h.quantity for h in balance_map.values())
        if balance_map
        else None
    )
    bal_change_1d_pct = None
    if bal_eval is not None and bal_change_1d_won is not None:
        yesterday_eval = bal_eval - bal_change_1d_won
        if yesterday_eval > 0:
            bal_change_1d_pct = bal_change_1d_won / yesterday_eval * 100.0

    return HourlyBalance(
        as_of=pd.Timestamp.now(tz="Asia/Seoul"),
        holdings=rows,
        balance_total_eval=bal_eval,
        balance_total_cost=bal_cost,
        balance_total_profit=bal_profit,
        balance_total_profit_pct=bal_profit_pct,
        balance_change_1d_won=bal_change_1d_won,
        balance_change_1d_pct=bal_change_1d_pct,
    )


def render_hourly_balance(hb: HourlyBalance) -> str:
    """장중 매시간 알림 — 통합 잔고 → 종목별 2줄 (이름 + 수량/매입/누적/오늘)."""
    weekday = ("월", "화", "수", "목", "금", "토", "일")[hb.as_of.weekday()]
    lines = [f"{hb.as_of.strftime('%Y-%m-%d')} ({weekday}) {hb.as_of.strftime('%H:%M')}"]

    # 통합 잔고 — 가장 위
    lines.append("")
    lines.append(_hdr("통합 잔고"))
    if hb.balance_total_eval is None:
        lines.append("KIS 잔고 미연동")
    else:
        eval_str = _fmt_won_compact(hb.balance_total_eval)
        cost_str = _fmt_won_compact(hb.balance_total_cost)
        lines.append(f"평가 {eval_str}원 (매입 {cost_str})")
        parts = []
        if hb.balance_change_1d_pct is not None:
            p = f"오늘 {hb.balance_change_1d_pct:+.2f}%"
            if hb.balance_change_1d_won is not None:
                p += f" ({_fmt_signed_won(hb.balance_change_1d_won)})"
            parts.append(p)
        if hb.balance_total_profit_pct is not None:
            p = f"누적 {hb.balance_total_profit_pct:+.2f}%"
            if hb.balance_total_profit is not None:
                p += f" ({_fmt_signed_won(hb.balance_total_profit)})"
            parts.append(p)
        if parts:
            lines.append(" / ".join(parts))

    # 보유 손익 — 종목별 2줄 (이름 / 수량·매입·누적·오늘)
    lines.append("")
    lines.append(_hdr("보유 손익"))
    if not hb.holdings:
        lines.append("보유 종목 없음")
    else:
        for h in sorted(hb.holdings, key=lambda x: -(x.profit_pct or 0)):
            shares = h.actual_shares or 0
            cost_won = (h.avg_price or 0) * shares
            cost_str = f"매입 {cost_won / 10000:,.1f}만"
            pp_str = f"{h.profit_pct:+.1f}%" if h.profit_pct is not None else "—"
            today_str = f"오늘 {h.ret_1d * 100:+.2f}%" if h.ret_1d is not None else "오늘 —"
            lines.append(f"{h.name}({shares}주/{cost_str})")
            lines.append(f": {pp_str}({today_str})")

    return "\n".join(lines)


def render_dual_alert(dual: DualSignal) -> str:
    """Compact 스타일 통합 알림.

    1메시지 + 섹션 분리. 이모지 최소화. 행동 상단. 포맷 일관.
    """
    cfg = dual.dual_config
    weekday = ("월", "화", "수", "목", "금", "토", "일")[dual.as_of.weekday()]
    lines: list[str] = [f"{dual.as_of} ({weekday})"]

    lines.extend(_render_top_actions(dual))
    lines.extend(_render_market_section(dual.market_state, cfg.ma_window))
    lines.extend(_render_balance_section_compact(dual))

    champ_pct = int(cfg.champion_pct * 100)
    lines.extend(
        _render_strategy_section(
            "챔피언",
            champ_pct,
            f"월간 {cfg.champion_lookback}m Top-{cfg.champion_top_n} + MA{cfg.ma_window}",
            dual.champion,
        )
    )

    agg_pct = int(cfg.aggressive_pct * 100)
    lines.extend(
        _render_strategy_section(
            "어그레시브",
            agg_pct,
            f"주간 {cfg.aggressive_lookback}m Top-{cfg.aggressive_top_n} + MA{cfg.ma_window}",
            dual.aggressive,
            warn="MDD 높음",
        )
    )

    lines.extend(_render_combined_actions(dual))
    lines.extend(_render_extremes_section(dual))

    # 빈 줄 정리 — 연속 빈 줄 1줄로 압축
    out: list[str] = []
    prev_empty = False
    for line in lines:
        if line.strip() == "":
            if prev_empty:
                continue
            prev_empty = True
        else:
            prev_empty = False
        out.append(line)
    return "\n".join(out)
