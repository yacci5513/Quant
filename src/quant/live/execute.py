"""시그널 → KIS API 자동 매매 실행 (듀얼 포트폴리오).

흐름:
1. 듀얼 시그널 계산 (챔피언 70% + 어그레시브 30%)
2. 현재 KIS 잔고 조회 (시그널 계산에 주입 → 중복 호출 회피)
3. 통합 매도/매수 액션 추출 (combined_sells / combined_buys)
4. KIS API로 주문
5. 텔레그램 결과 발송 (compact 듀얼 알림 + 주문 결과)

가드레일 강제:
§3 미래참조: signal 생성 시 자동 보호 (engine.auto_shift_weights)
§9 Kill switch: 포트 누적 손익률 ≤ KILL_SWITCH_PCT 시 전량 매도
§10 라이브 진입 룰: SEED_WON 미설정 시 차단 + 시드 분할로 가중 MDD 제어
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from tenacity import RetryError

from quant.backtest.risk import (
    ExposureLimits,
    KillSwitchConfig,
    pre_trade_check,
)
from quant.common.config import get_settings
from quant.common.logger import logger
from quant.daily_signal import (
    AlertType,
    DualSignalConfig,
    compute_dual_signal,
    render_dual_alert,
)
from quant.live.client import (
    KISError,
    OrderResult,
    get_balance,
    get_quote,
    order_cash,
    round_to_tick,
)
from quant.live.killswitch import evaluate as eval_killswitch
from quant.live.killswitch import save_state as save_kill_state
from quant.live.trailing import (
    check_trail_trigger,
    load_trailing,
    prune_stale,
    save_trailing,
    update_max,
)
from quant.notify.telegram import send_telegram

# 지정가 매수 시 어제 종가 대비 허용 프리미엄 (슬리피지 방지용 상한)
BUY_LIMIT_PREMIUM = 0.005  # +0.5%

# 종목별 손절·익절 한도 (KIS 평가손익률 기준, %)
# 2026-05-18: -10/+20 → -15/+40 완화 (시장 강세 시 알파 죽이는 문제 + 단발 노이즈 손절 회피)
STOP_LOSS_PCT = -15.0  # -15% 도달 시 손절
TAKE_PROFIT_PCT = 40.0  # +40% 도달 시 익절 (추세 가드 +5%로 더 보류 가능)

# 포트폴리오 누적 손실 Kill switch (%) — 2일 연속 위반 시 발동, 손실 종목만 매도
# (단발 노이즈로 +20% 종목까지 던지는 문제 해결, 2026-05-17)
KILL_SWITCH_PCT = -20.0

# 당일 추세 가드 (당일 변동률 %)
# 손절 조건이지만 당일 +TREND_GUARD_STOP% 이상 회복 → 손절 보류
# 익절 조건이지만 당일 +TREND_GUARD_PROFIT% 이상 상승 → 익절 보류 (더 갈 수도)
TREND_GUARD_STOP = 2.0
TREND_GUARD_PROFIT = 5.0

# tenacity가 KISError를 재시도 후 RetryError로 wrap해서 던지므로 둘 다 잡아야 함
_OrderExc = (KISError, RetryError)


@dataclass
class ExecutionResult:
    sells: list[OrderResult]
    buys: list[OrderResult]
    skipped: list[str]
    notes: list[str]


def _check_seed() -> int:
    """SEED_WON 검증 — 0이면 매매 차단."""
    s = get_settings()
    if s.seed_won <= 0:
        raise RuntimeError("SEED_WON 미설정 — .env에 SEED_WON=5000000 같이 설정 필요. 매매 차단.")
    return s.seed_won


def _derive_overall_alert(dual) -> AlertType:
    """듀얼 시그널에서 통합 알림 타입 추론.

    LIQUIDATE: 양쪽 모두 LIQUIDATE (시장 약세 진입)
    REBALANCE: 통합 매매 액션이 1개 이상 존재
    RECOVER  : 둘 중 하나라도 RECOVER (다른 쪽은 NORMAL)
    NORMAL   : 그 외 (보유 유지)
    """
    c, a = dual.champion.alert_type, dual.aggressive.alert_type
    if c is AlertType.LIQUIDATE and a is AlertType.LIQUIDATE:
        return AlertType.LIQUIDATE
    if dual.combined_sells or dual.combined_buys:
        return AlertType.REBALANCE
    if AlertType.RECOVER in (c, a):
        return AlertType.RECOVER
    return AlertType.NORMAL


def execute_rebalance(
    *,
    dry_run: bool = True,
    seed_override: int | None = None,
    notify: bool = True,
    aggressive_pct: float = 0.30,
) -> ExecutionResult:
    """듀얼 시그널 기반 자동 리밸런싱.

    챔피언(월간 12m+MA100, 70%) + 어그레시브(주간 1m Top-5+MA100, 30%)
    두 시그널 합집합 - 잔고 = 통합 매도/매수 액션.

    Args:
        dry_run: True면 주문 실제 호출 X (시뮬레이션, 메시지만 발송)
        seed_override: 시드 강제 지정 (None이면 settings.seed_won)
        notify: 텔레그램 발송 여부
        aggressive_pct: 어그레시브 시드 비중 (0.0~1.0, 기본 0.30)
    """
    seed = seed_override if seed_override is not None else _check_seed()
    s = get_settings()

    # 1. 현재 보유 먼저 조회 → 시그널에 매입가/손익 주입 (잔고 fetch 1회)
    holdings = get_balance()
    held_map = {h.ticker: h for h in holdings}

    # 2. 듀얼 시그널 계산
    dual_cfg = DualSignalConfig(
        champion_pct=1.0 - aggressive_pct,
        aggressive_pct=aggressive_pct,
    )
    dual = compute_dual_signal(
        seed_won=float(seed),
        dual_config=dual_cfg,
        fetch_balance=False,
        balance_map=held_map,
    )
    notes_combined = list(dual.champion.notes) + list(dual.aggressive.notes)

    sells_orders: list[OrderResult] = []
    buys_orders: list[OrderResult] = []
    skipped: list[str] = []
    notes: list[str] = notes_combined
    notes.append(
        f"모드: {s.kis_mode.value} / 시드: {seed:,}원 "
        f"(챔피언 {int(dual_cfg.champion_pct * 100)}% / 어그 {int(dual_cfg.aggressive_pct * 100)}%) "
        f"/ dry_run: {dry_run}"
    )

    def _dry_order(ticker: str, qty: int, side: str, est_price: float | None) -> OrderResult:
        """dry-run 가짜 OrderResult — render에서 그대로 종목명·가격 표시 가능."""
        return OrderResult(
            order_no="DRY",
            ticker=ticker,
            side=side,
            quantity=qty,
            requested_price=est_price,
            success=True,
            raw={},
        )

    def _today_change_pct(ticker: str) -> float:
        """장중 흐름 (오늘 시가 → 현재가 %, 갭 분리). 실패 시 0.

        예) 시가 95,000 + 현재 96,000 → +1.05%. 어제 종가 대비가 아니라
        장중 회복 추세를 정확히 잡기 위함.
        """
        try:
            q = get_quote(ticker)
            if q.open_price > 0:
                return (q.price - q.open_price) / q.open_price * 100.0
            return q.change_pct  # 시가 0 (장 시작 전 등) fallback
        except Exception:
            return 0.0

    def _sell_all(reason_per_ticker: dict[str, str]) -> None:
        """주어진 종목 즉시 시장가 전량 매도. forced 동작."""
        for ticker, reason in reason_per_ticker.items():
            h = held_map.get(ticker)
            if not h:
                continue
            notes.append(reason)
            if dry_run:
                sells_orders.append(_dry_order(ticker, h.quantity, "sell", h.current_price))
                continue
            try:
                r = order_cash(ticker=ticker, quantity=h.quantity, side="sell", price=None)
                sells_orders.append(r)
            except _OrderExc as e:
                skipped.append(f"매도 실패 {ticker}: {e}")

    # 2a. 포트폴리오 Kill switch — 2일 연속 임계 위반 시 발동, 손실 종목만 매도
    port_pp = dual.balance_total_profit_pct
    from datetime import date as _date

    decision, new_kill_state = eval_killswitch(_date.today(), port_pp, threshold=KILL_SWITCH_PCT)
    save_kill_state(new_kill_state)
    if decision.armed and not decision.should_fire:
        notes.append(decision.reason)  # 1일째 armed — 알림만, 매도 X
    if decision.should_fire:
        # 손실 종목(profit_pct < 0)만 매도. 이익 종목은 보존.
        losers = {
            h.ticker: f"Kill switch {h.ticker} {h.profit_pct:+.1f}%"
            for h in holdings
            if h.profit_pct is not None and h.profit_pct < 0
        }
        keepers = [h for h in holdings if h.profit_pct is not None and h.profit_pct >= 0]
        notes.append(decision.reason)
        notes.append(
            f"손실 종목 {len(losers)}건만 매도 (이익 종목 {len(keepers)}건 보존: "
            f"{', '.join(h.ticker for h in keepers[:5])}{'...' if len(keepers) > 5 else ''})"
        )
        _sell_all(losers)
        if notify:
            msg = render_execution_result(
                ExecutionResult(sells=sells_orders, buys=buys_orders, skipped=skipped, notes=notes),
                signal_type=AlertType.KILL_SWITCH,
                dry_run=dry_run,
            )
            try:
                send_telegram(msg, monospace=True)
            except Exception as e:
                logger.warning(f"텔레그램 발송 실패: {e}")
        return ExecutionResult(sells=sells_orders, buys=buys_orders, skipped=skipped, notes=notes)

    # 2b. 종목별 손절·익절·트레일링 (당일 추세 가드 포함)
    # 트레일링 데이터 로드 + 보유 종목 max_profit_pct 갱신
    trail = load_trailing()
    prune_stale(trail, {h.ticker for h in holdings})
    for h in holdings:
        if h.profit_pct is not None:
            update_max(h.ticker, h.profit_pct, trail)

    forced_sells: set[str] = set()
    for h in holdings:
        pp = h.profit_pct
        if pp is None:
            continue
        hit_stop = pp <= STOP_LOSS_PCT
        hit_take = pp >= TAKE_PROFIT_PCT
        hit_trail, max_seen = check_trail_trigger(h.ticker, pp, trail)
        if not (hit_stop or hit_take or hit_trail):
            continue
        today_chg = _today_change_pct(h.ticker)
        # 손절: 당일 +X% 회복 추세면 보류
        if hit_stop and not (hit_take or hit_trail) and today_chg >= TREND_GUARD_STOP:
            notes.append(f"⏸ {h.ticker} 손절 보류 ({pp:+.1f}%, 당일 {today_chg:+.1f}% 회복 추세)")
            continue
        # 익절: 당일 +Y% 상승 추세면 보류 (더 갈 수도)
        if hit_take and not hit_trail and today_chg >= TREND_GUARD_PROFIT:
            notes.append(f"⏸ {h.ticker} 익절 보류 ({pp:+.1f}%, 당일 {today_chg:+.1f}% 상승 추세)")
            continue
        # 트레일링: 당일 회복 추세 가드 동일 적용 (당일 +X% 회복 시 보류)
        if hit_trail and not hit_take and today_chg >= TREND_GUARD_STOP:
            notes.append(
                f"⏸ {h.ticker} 트레일링 보류 (최고 {max_seen:+.1f}% → {pp:+.1f}%, "
                f"당일 {today_chg:+.1f}% 회복)"
            )
            continue
        if hit_trail:
            reason = f"📉 트레일링 (최고 {max_seen:+.1f}% → 현재 {pp:+.1f}%)"
        elif hit_stop:
            reason = "💰 손절"
        else:
            reason = "🎯 익절"
        notes.append(f"{reason} {h.ticker} {pp:+.1f}% (당일 {today_chg:+.1f}%)")
        if dry_run:
            sells_orders.append(_dry_order(h.ticker, h.quantity, "sell", h.current_price))
            forced_sells.add(h.ticker)
            continue
        try:
            r = order_cash(ticker=h.ticker, quantity=h.quantity, side="sell", price=None)
            sells_orders.append(r)
            forced_sells.add(h.ticker)
        except _OrderExc as e:
            skipped.append(f"매도 실패 {h.ticker}: {e}")

    # 트레일링 데이터 저장 (갱신된 max_profit_pct)
    try:
        save_trailing(trail)
    except Exception as e:
        logger.warning(f"trailing.json 저장 실패: {e}")

    # 2c. 듀얼 통합 매매 액션 — combined_sells / combined_buys 직접 사용
    # LIQUIDATE 시 해당 시그널 holdings=[] → 자연스럽게 combined_sells에 흡수됨
    # NORMAL이면 combined_sells/buys 모두 빈 리스트
    sell_tickers = {h.ticker for h in dual.combined_sells}
    for ticker in sell_tickers - forced_sells:
        h = held_map.get(ticker)
        if not h:
            continue
        if dry_run:
            sells_orders.append(_dry_order(ticker, h.quantity, "sell", h.current_price))
            continue
        try:
            r = order_cash(ticker=ticker, quantity=h.quantity, side="sell", price=None)
            sells_orders.append(r)
        except _OrderExc as e:
            skipped.append(f"매도 실패 {ticker}: {e}")

    # 매수: combined_buys (지정가 = 어제 종가 × (1 + BUY_LIMIT_PREMIUM), 호가 단위 라운드)
    for buy in dual.combined_buys:
        if not buy.target_shares or buy.target_shares <= 0:
            continue
        if buy.ticker in forced_sells:
            continue  # 손절·익절로 매도된 종목은 같은 날 재매수 안 함
        actual_qty = held_map.get(buy.ticker)
        need = buy.target_shares - (actual_qty.quantity if actual_qty else 0)
        if need <= 0:
            continue
        limit_price = None
        if buy.last_close:
            limit_price = round_to_tick(buy.last_close * (1 + BUY_LIMIT_PREMIUM))
        if dry_run:
            buys_orders.append(_dry_order(buy.ticker, need, "buy", limit_price))
            continue
        try:
            r = order_cash(ticker=buy.ticker, quantity=need, side="buy", price=limit_price)
            buys_orders.append(r)
        except _OrderExc as e:
            skipped.append(f"매수 실패 {buy.ticker}: {e}")

    # 알림: 듀얼 시그널 본문(시장·잔고·전략별·매매상세) + 실행 결과 요약 추가
    if notify:
        derived_type = _derive_overall_alert(dual)
        dual_text = render_dual_alert(dual)
        exec_text = render_execution_result(
            ExecutionResult(sells=sells_orders, buys=buys_orders, skipped=skipped, notes=notes),
            signal_type=derived_type,
            dry_run=dry_run,
        )
        msg = dual_text + "\n\n" + exec_text
        try:
            send_telegram(msg, monospace=True)
        except Exception as e:
            logger.warning(f"텔레그램 발송 실패: {e}")

    return ExecutionResult(sells=sells_orders, buys=buys_orders, skipped=skipped, notes=notes)


def _name_map() -> dict[str, str]:
    """종목번호 → 한글명 매핑 (KOSPI + KOSDAQ)."""
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


def render_execution_result(res: ExecutionResult, *, signal_type: AlertType, dry_run: bool) -> str:
    """실행 결과 → 텔레그램 메시지 (종목명 + 가격 포함)."""
    icon = "🧪 DRY-RUN" if dry_run else "🤖 실거래 체결"
    lines = [f"{icon} | {signal_type.value}"]
    names = _name_map()

    def _fmt_order(o: OrderResult) -> str:
        nm = names.get(o.ticker, "?")
        # raw 응답에서 체결가 시도 (실거래 시), 없으면 requested_price 또는 시장가
        price_part = ""
        if o.requested_price:
            price_part = f" @{int(o.requested_price):,}원"
        elif isinstance(o.raw, dict):
            output = o.raw.get("output", {}) if isinstance(o.raw.get("output"), dict) else {}
            price = output.get("AVG_PRVS") or output.get("ord_unpr")
            if price:
                with contextlib.suppress(ValueError, TypeError):
                    price_part = f" @{int(float(price)):,}원"
            else:
                price_part = " (시장가)"
        else:
            price_part = " (시장가)"
        order_part = f" #{o.order_no}" if o.order_no else ""
        return f"  · {o.ticker} {nm} {o.quantity}주{price_part}{order_part}"

    if res.sells:
        lines.append("")
        lines.append(f"🔴 매도 {len(res.sells)}건:")
        for o in res.sells:
            lines.append(_fmt_order(o))

    if res.buys:
        lines.append("")
        lines.append(f"🟢 매수 {len(res.buys)}건:")
        for o in res.buys:
            lines.append(_fmt_order(o))

    if res.skipped:
        lines.append("")
        lines.append(f"⚠️ Skip/실패 {len(res.skipped)}건:")
        for s in res.skipped[:10]:
            enriched = s
            for code, nm in names.items():
                if code in s:
                    enriched = s.replace(code, f"{code} {nm}", 1)
                    break
            lines.append(f"  · {enriched}")
        if len(res.skipped) > 10:
            lines.append(f"  ... +{len(res.skipped) - 10}건 더")

    if res.notes:
        lines.append("")
        lines.append("📝 정보:")
        for n in res.notes:
            lines.append(f"  · {n}")

    return "\n".join(lines)


def _quietly_use_get_quote() -> None:
    """linter — get_quote는 향후 한도 검증에 사용 예정."""
    _ = get_quote
    _ = pre_trade_check
    _ = ExposureLimits
    _ = KillSwitchConfig
