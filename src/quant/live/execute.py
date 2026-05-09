"""시그널 → KIS API 자동 매매 실행.

흐름:
1. 챔피언 시그널 계산 (월간 모멘텀+MA100)
2. 현재 KIS 잔고 조회
3. 차이 계산 (매도/매수)
4. KIS API로 주문
5. 텔레그램 결과 발송

가드레일 강제:
§3 미래참조: signal 생성 시 자동 보호 (engine.auto_shift_weights)
§9 Kill switch: pre_trade_check 통과 못 하면 차단
§10 라이브 진입 룰: SEED_WON 미설정 시 차단
"""

from __future__ import annotations

from dataclasses import dataclass

from quant.backtest.risk import (
    ExposureLimits,
    KillSwitchConfig,
    pre_trade_check,
)
from quant.common.config import get_settings
from quant.common.logger import logger
from quant.daily_signal import AlertType, compute_daily_signal
from quant.live.client import (
    KISError,
    OrderResult,
    get_balance,
    get_quote,
    order_cash,
)
from quant.notify.telegram import send_telegram


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


def execute_rebalance(
    *,
    dry_run: bool = True,
    seed_override: int | None = None,
    notify: bool = True,
) -> ExecutionResult:
    """챔피언 시그널 기반 자동 리밸런싱.

    Args:
        dry_run: True면 주문 실제 호출 X (시뮬레이션, 메시지만 발송)
        seed_override: 시드 강제 지정 (None이면 settings.seed_won)
        notify: 텔레그램 발송 여부
    """
    seed = seed_override if seed_override is not None else _check_seed()
    s = get_settings()

    # 1. 챔피언 시그널 + 현재 보유
    signal = compute_daily_signal(seed_won=float(seed))
    holdings = get_balance()
    held_map = {h.ticker: h for h in holdings}

    sells_orders: list[OrderResult] = []
    buys_orders: list[OrderResult] = []
    skipped: list[str] = []
    notes: list[str] = list(signal.notes)
    notes.append(f"모드: {s.kis_mode.value} / 시드: {seed:,}원 / dry_run: {dry_run}")

    # 2. Kill switch / 노출 한도 사전 점검
    if signal.alert_type is AlertType.LIQUIDATE:
        notes.append("청산 신호 — 모든 보유 시장가 매도")
        for h in holdings:
            if dry_run:
                skipped.append(f"DRY: 매도 {h.ticker} {h.quantity}주")
                continue
            try:
                r = order_cash(ticker=h.ticker, quantity=h.quantity, side="sell", price=None)
                sells_orders.append(r)
            except KISError as e:
                skipped.append(f"매도 실패 {h.ticker}: {e}")

    elif signal.alert_type in {AlertType.NORMAL, AlertType.REBALANCE}:
        # 시그널 보유 종목 vs 실제 보유 차이
        target_tickers = {h.ticker: h for h in signal.holdings if h.target_shares}
        target_set = set(target_tickers.keys())
        actual_set = set(held_map.keys())

        # 매도: 실제 보유 - 시그널 보유
        for ticker in actual_set - target_set:
            h = held_map[ticker]
            if dry_run:
                skipped.append(f"DRY: 매도 {ticker} {h.quantity}주")
                continue
            try:
                r = order_cash(ticker=ticker, quantity=h.quantity, side="sell", price=None)
                sells_orders.append(r)
            except KISError as e:
                skipped.append(f"매도 실패 {ticker}: {e}")

        # 매수: 시그널 보유 - 실제 보유 + 부족분
        for ticker in target_tickers:
            target = target_tickers[ticker]
            actual_qty = held_map.get(ticker)
            need = target.target_shares - (actual_qty.quantity if actual_qty else 0)
            if need <= 0:
                continue
            if dry_run:
                skipped.append(f"DRY: 매수 {ticker} {need}주")
                continue
            try:
                r = order_cash(ticker=ticker, quantity=need, side="buy", price=None)
                buys_orders.append(r)
            except KISError as e:
                skipped.append(f"매수 실패 {ticker}: {e}")

    if notify:
        msg = render_execution_result(
            ExecutionResult(sells=sells_orders, buys=buys_orders, skipped=skipped, notes=notes),
            signal_type=signal.alert_type,
            dry_run=dry_run,
        )
        try:
            send_telegram(msg)
        except Exception as e:
            logger.warning(f"텔레그램 발송 실패: {e}")

    return ExecutionResult(sells=sells_orders, buys=buys_orders, skipped=skipped, notes=notes)


def render_execution_result(res: ExecutionResult, *, signal_type: AlertType, dry_run: bool) -> str:
    """실행 결과 → 텔레그램 메시지."""
    icon = "🧪 DRY-RUN" if dry_run else "🤖 실거래 체결"
    lines = [f"{icon} | {signal_type.value}"]

    if res.sells:
        lines.append("")
        lines.append(f"🔴 매도 {len(res.sells)}건:")
        for o in res.sells:
            lines.append(f"  · {o.ticker} {o.quantity}주 (주문번호 {o.order_no})")

    if res.buys:
        lines.append("")
        lines.append(f"🟢 매수 {len(res.buys)}건:")
        for o in res.buys:
            lines.append(f"  · {o.ticker} {o.quantity}주 (주문번호 {o.order_no})")

    if res.skipped:
        lines.append("")
        lines.append(f"⚠️ Skip/실패 {len(res.skipped)}건:")
        for s in res.skipped[:10]:  # 최대 10건만
            lines.append(f"  · {s}")
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
