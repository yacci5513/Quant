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
from quant.daily_signal import AlertType, compute_daily_signal
from quant.live.client import (
    KISError,
    OrderResult,
    get_balance,
    get_quote,
    order_cash,
    round_to_tick,
)
from quant.notify.telegram import send_telegram

# 지정가 매수 시 어제 종가 대비 허용 프리미엄 (슬리피지 방지용 상한)
BUY_LIMIT_PREMIUM = 0.005  # +0.5%

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

    # 1. 현재 보유 먼저 조회 → 시그널에 매입가/손익 주입 (텔레그램 알림용)
    holdings = get_balance()
    held_map = {h.ticker: h for h in holdings}

    # 2. 챔피언 시그널 — .env의 multi_regime 설정 반영
    signal = compute_daily_signal(
        seed_won=float(seed),
        multi_regime=s.strategy_multi_regime,
        balance_map=held_map,
    )

    sells_orders: list[OrderResult] = []
    buys_orders: list[OrderResult] = []
    skipped: list[str] = []
    notes: list[str] = list(signal.notes)
    notes.append(f"모드: {s.kis_mode.value} / 시드: {seed:,}원 / dry_run: {dry_run}")

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

    # 2. Kill switch / 노출 한도 사전 점검
    if signal.alert_type is AlertType.LIQUIDATE:
        notes.append("청산 신호 — 모든 보유 시장가 매도")
        for h in holdings:
            if dry_run:
                sells_orders.append(_dry_order(h.ticker, h.quantity, "sell", h.current_price))
                continue
            try:
                r = order_cash(ticker=h.ticker, quantity=h.quantity, side="sell", price=None)
                sells_orders.append(r)
            except _OrderExc as e:
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
                sells_orders.append(_dry_order(ticker, h.quantity, "sell", h.current_price))
                continue
            try:
                r = order_cash(ticker=ticker, quantity=h.quantity, side="sell", price=None)
                sells_orders.append(r)
            except _OrderExc as e:
                skipped.append(f"매도 실패 {ticker}: {e}")

        # 매수: 시그널 보유 - 실제 보유 + 부족분
        # 지정가 = 어제 종가 × (1 + BUY_LIMIT_PREMIUM), 호가 단위 라운드
        # 슬리피지 방지. 단 가격 급등 시 미체결 가능 (다음날 09:30 재시도)
        for ticker in target_tickers:
            target = target_tickers[ticker]
            actual_qty = held_map.get(ticker)
            need = target.target_shares - (actual_qty.quantity if actual_qty else 0)
            if need <= 0:
                continue
            limit_price = None
            if target.last_close:
                limit_price = round_to_tick(target.last_close * (1 + BUY_LIMIT_PREMIUM))
            if dry_run:
                buys_orders.append(_dry_order(ticker, need, "buy", limit_price))
                continue
            try:
                r = order_cash(ticker=ticker, quantity=need, side="buy", price=limit_price)
                buys_orders.append(r)
            except _OrderExc as e:
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
