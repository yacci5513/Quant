"""실거래 직전 시그널 추출.

저장된 가격 패널에서 최신 데이터로 시그널을 계산하고
"오늘 매수해야 할 종목 N개 + 가중치 + 종목명" 을 출력한다.

가드레일:
- 시그널 계산은 최근 영업일 기준 (오늘 데이터를 알 수 없으므로 .shift(1) 자동 적용)
- 시드 금액 입력 시 종목별 매수 금액 + 주식 수 추정 (호가 단위 미반영)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant.common.logger import logger
from quant.data.price.fetch_krx import load_close_panel, load_value_panel
from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights


@dataclass
class SignalRow:
    ticker: str
    name: str
    weight: float
    last_close: float
    momentum_pct: float  # 시그널 점수 = 모멘텀 수익률
    target_value_won: float | None = None  # 시드가 주어졌을 때
    target_shares: int | None = None


def _ticker_name_map() -> dict[str, str]:
    """종목 코드 → 한글명. fdr StockListing 1회 호출."""
    try:
        import FinanceDataReader as fdr  # noqa: N813

        listing = fdr.StockListing("KOSPI")
        return dict(zip(listing["Code"], listing["Name"], strict=False))
    except Exception as e:
        logger.warning(f"종목명 조회 실패 (네트워크 이슈?): {e}")
        return {}


def latest_momentum_signal(
    config: MomentumTopNConfig | None = None,
    seed_won: float | None = None,
) -> list[SignalRow]:
    """최신 영업일 기준 모멘텀 Top-N 시그널.

    seed_won: 투자 자본 (원). 입력 시 종목별 목표 매수 금액·주식 수 계산.
    """
    cfg = config or MomentumTopNConfig()
    prices = load_close_panel()
    values = load_value_panel()
    if prices.empty:
        raise RuntimeError("저장된 가격 데이터 없음 — quant data fetch-krx 먼저 실행")

    weights_panel = generate_weights(prices, values=values, config=cfg)

    # 마지막 리밸런싱 시점의 가중치
    rb_mask = weights_panel.sum(axis=1) > 0
    if not rb_mask.any():
        raise RuntimeError("리밸런싱 데이터 없음 — 데이터 부족")
    last_rb_date = weights_panel.index[rb_mask][-1]
    selected = weights_panel.loc[last_rb_date]
    selected = selected[selected > 0].sort_values(ascending=False)

    # 시그널 점수 (12-1 모멘텀)
    end = prices.shift(cfg.skip_months * 21)
    start = prices.shift(cfg.lookback_months * 21)
    momentum = end / start - 1.0

    name_map = _ticker_name_map()
    last_close = prices.loc[last_rb_date]
    last_signal_date = prices.index[-1].date()

    rows: list[SignalRow] = []
    for ticker, w in selected.items():
        close = float(last_close.get(ticker, float("nan")))
        mom = float(momentum.loc[last_rb_date].get(ticker, float("nan")))
        target_value = seed_won * w if seed_won else None
        target_shares = int(target_value // close) if target_value and close > 0 else None
        rows.append(
            SignalRow(
                ticker=ticker,
                name=name_map.get(ticker, "?"),
                weight=float(w),
                last_close=close,
                momentum_pct=mom,
                target_value_won=target_value,
                target_shares=target_shares,
            )
        )

    logger.info(f"마지막 리밸런싱: {last_rb_date.date()}, 데이터 최신일: {last_signal_date}")
    return rows


def render_signal_table(rows: list[SignalRow], seed_won: float | None = None) -> str:
    if not rows:
        return "(시그널 없음)"
    has_seed = seed_won is not None and rows[0].target_value_won is not None
    lines = []
    if has_seed:
        header = f"{'#':>3}  {'종목코드':<8} {'종목명':<14} {'가중치':>7} {'종가':>10} {'모멘텀':>9} {'매수금액':>14} {'수량':>8}"
    else:
        header = (
            f"{'#':>3}  {'종목코드':<8} {'종목명':<14} {'가중치':>7} {'종가':>10} {'모멘텀':>9}"
        )
    lines.append(header)
    lines.append("-" * len(header))
    for i, r in enumerate(rows, 1):
        if has_seed:
            lines.append(
                f"{i:>3}  {r.ticker:<8} {r.name:<14} {r.weight * 100:>6.2f}%  "
                f"{r.last_close:>10,.0f}  {r.momentum_pct * 100:>+8.2f}%  "
                f"{(r.target_value_won or 0):>14,.0f}원 {(r.target_shares or 0):>7}주"
            )
        else:
            lines.append(
                f"{i:>3}  {r.ticker:<8} {r.name:<14} {r.weight * 100:>6.2f}%  "
                f"{r.last_close:>10,.0f}  {r.momentum_pct * 100:>+8.2f}%"
            )
    if has_seed:
        total = sum(r.target_value_won or 0 for r in rows)
        cash = (seed_won or 0) - total
        lines.append("-" * len(header))
        lines.append(f"{'합계':>30}  {total:>14,.0f}원   (현금 {cash:>+,.0f}원)")
    return "\n".join(lines)


def _ignore() -> None:
    _ = (date, pd)
