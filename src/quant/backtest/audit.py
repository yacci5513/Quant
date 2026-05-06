"""백테스트 거래 이력 감사 (rebalance log).

매 리밸런싱 시점에 어떤 종목을 보유했고, 다음 리밸런싱까지 얼마나 수익이 났는지 기록.
"이 전략을 5년 전부터 굴렸으면 매월 어떤 결정이었나"를 한눈에 확인.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant.common.logger import logger


def build_rebalance_log(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """리밸런싱 시점별 보유 종목 + 다음 리밸런싱까지 수익률 표.

    Columns: rebalance_date, next_rebalance_date, ticker, weight, entry_close,
             exit_close, return_pct, contribution_to_portfolio
    """
    # 리밸런싱 시점 = 가중치가 변하는 날 (forward-fill된 사이는 제외)
    diff = weights.diff().abs().sum(axis=1)
    # 첫째 날 + 변화가 있는 날
    rb_dates = weights.index[(diff > 1e-6) | (diff.index == weights.index[0])].tolist()
    # 첫 진입이 0이면 제외
    rb_dates = [d for d in rb_dates if weights.loc[d].sum() > 0]
    rows = []
    for i, d in enumerate(rb_dates):
        next_d = rb_dates[i + 1] if i + 1 < len(rb_dates) else weights.index[-1]
        positions = weights.loc[d]
        held = positions[positions > 0]
        for ticker, w in held.items():
            entry = prices.loc[d, ticker] if ticker in prices.columns else None
            exit_ = prices.loc[next_d, ticker] if ticker in prices.columns else None
            if entry is None or exit_ is None or pd.isna(entry) or pd.isna(exit_):
                ret = float("nan")
                contrib = float("nan")
            else:
                ret = float(exit_ / entry - 1.0)
                contrib = float(w) * ret
            rows.append(
                {
                    "rebalance_date": d.date(),
                    "next_rebalance_date": next_d.date(),
                    "ticker": ticker,
                    "weight": float(w),
                    "entry_close": float(entry) if entry is not None else None,
                    "exit_close": float(exit_) if exit_ is not None else None,
                    "return_pct": ret,
                    "contribution": contrib,
                }
            )
    return pd.DataFrame(rows)


def summarize_log(log: pd.DataFrame) -> pd.DataFrame:
    """종목별 누적 통계 (등장 횟수 / 평균 수익률 / 기여도)."""
    by_ticker = log.groupby("ticker").agg(
        appearances=("ticker", "count"),
        avg_return=("return_pct", "mean"),
        win_rate=("return_pct", lambda s: float((s > 0).mean()) if len(s) else 0.0),
        total_contribution=("contribution", "sum"),
    )
    return by_ticker.sort_values("total_contribution", ascending=False)


def save_log(out_dir: Path, prices: pd.DataFrame, weights: pd.DataFrame) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = build_rebalance_log(prices, weights)
    by_ticker = summarize_log(log)

    p_log = out_dir / "rebalance_log.csv"
    p_top = out_dir / "ticker_summary.csv"
    log.to_csv(p_log, index=False)
    by_ticker.to_csv(p_top)

    # Top contributors / detractors
    if len(by_ticker) > 0:
        logger.info("\n=== Top 10 Contributors ===")
        for ticker, row in by_ticker.head(10).iterrows():
            logger.info(
                f"  {ticker}  등장 {int(row['appearances']):>2}회  "
                f"평균 {row['avg_return'] * 100:>+6.2f}%  "
                f"승률 {row['win_rate'] * 100:>5.1f}%  "
                f"기여 {row['total_contribution'] * 100:>+6.2f}%"
            )
        logger.info("\n=== Bottom 5 Detractors ===")
        for ticker, row in by_ticker.tail(5).iterrows():
            logger.info(
                f"  {ticker}  등장 {int(row['appearances']):>2}회  "
                f"평균 {row['avg_return'] * 100:>+6.2f}%  "
                f"승률 {row['win_rate'] * 100:>5.1f}%  "
                f"기여 {row['total_contribution'] * 100:>+6.2f}%"
            )

    return {"log": p_log, "ticker_summary": p_top}
