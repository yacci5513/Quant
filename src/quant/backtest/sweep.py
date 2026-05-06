"""파라미터 그리드 sweep.

가드레일 §1: 파라미터 한 개만 바꿔도 결과가 무너지면 과적합 신호.
서로 가까운 파라미터끼리 비슷한 결과를 내야 진짜 알파.

(top_n, lookback) 조합에 대해 백테스트를 돌리고 Sharpe·CAGR·MDD 패널 반환.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from quant.backtest.costs import BLUECHIP_KIS, CostModel
from quant.backtest.engine import run_backtest
from quant.common.logger import logger
from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights


@dataclass
class SweepResult:
    """파라미터 그리드 결과 — long format DataFrame."""

    df: pd.DataFrame  # columns: top_n, lookback, sharpe, cagr, mdd, n_trades

    def pivot(self, value: str = "sharpe") -> pd.DataFrame:
        """heatmap용 wide format. index=top_n, columns=lookback."""
        return self.df.pivot(index="top_n", columns="lookback", values=value)

    def stability_score(self) -> float:
        """그리드 전체의 Sharpe 표준편차 / 평균 절댓값.

        낮을수록 견고. 1.0 이상이면 파라미터 의존성 너무 큼.
        """
        s = self.df["sharpe"]
        if abs(s.mean()) < 1e-6:
            return float("inf")
        return float(s.std(ddof=0) / abs(s.mean()))

    def report(self) -> str:
        s = self.df["sharpe"]
        c = self.df["cagr"]
        return (
            f"Parameter sweep ({len(self.df)} combos)\n"
            f"  Sharpe : {s.mean():+.2f} ± {s.std(ddof=0):.2f} "
            f"[min {s.min():+.2f}, max {s.max():+.2f}]\n"
            f"  CAGR   : {c.mean() * 100:+.2f}% ± {c.std(ddof=0) * 100:.2f}%\n"
            f"  견고성  : stability={self.stability_score():.2f} "
            f"(낮을수록 좋음, < 0.5 양호)\n"
        )


def run_sweep(
    prices: pd.DataFrame,
    *,
    values: pd.DataFrame | None = None,
    top_n_grid: list[int] | None = None,
    lookback_grid: list[int] | None = None,
    skip_months: int = 1,
    min_value: float = 1e9,
    cost_model: CostModel = BLUECHIP_KIS,
) -> SweepResult:
    """모든 (top_n, lookback) 조합 백테스트."""
    top_n_grid = top_n_grid or [5, 10, 20]
    lookback_grid = lookback_grid or [3, 6, 12]

    rows = []
    total = len(top_n_grid) * len(lookback_grid)
    for i, (n, lb) in enumerate(product(top_n_grid, lookback_grid), 1):
        cfg = MomentumTopNConfig(
            top_n=n,
            lookback_months=lb,
            skip_months=skip_months,
            min_avg_value=min_value,
        )
        w = generate_weights(prices, values=values, config=cfg)
        res = run_backtest(prices, w, cost_model=cost_model)
        m = res.metrics
        rows.append(
            {
                "top_n": n,
                "lookback": lb,
                "sharpe": m.sharpe,
                "cagr": m.cagr,
                "mdd": m.max_drawdown,
                "n_trades": m.n_trades,
                "turnover": m.turnover_annual or 0.0,
            }
        )
        logger.info(
            f"[{i}/{total}] top_n={n}, lookback={lb}m → "
            f"Sharpe {m.sharpe:+.2f}, CAGR {m.cagr * 100:+.2f}%, MDD {m.max_drawdown * 100:.1f}%"
        )

    df = pd.DataFrame(rows)
    out = SweepResult(df=df)
    logger.info(out.report())
    if out.stability_score() > 1.0:
        logger.warning(
            "⚠️ Stability > 1.0 — 파라미터에 매우 민감, 과적합 가능성. "
            "범위를 좁히거나 단순한 시그널로 후퇴."
        )
    return out
