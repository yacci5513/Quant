"""Walk-forward 검증.

가드레일 §1: 단일 IS/OOS 분리는 한 번 운 좋게 잘리면 통과 가능.
Walk-forward는 여러 시점 OOS를 평균 내서 안정성을 본다.

설계:
- 전체 가격 패널을 받아 rolling window별 백테스트 실행
- 각 윈도우의 OOS 결과 모음 → 분포 통계 (mean, std, min, max)
- 윈도우별 Sharpe·CAGR·MDD가 일관되게 좋아야 진짜 알파
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quant.backtest.costs import BLUECHIP_KIS, CostModel
from quant.backtest.engine import run_backtest
from quant.backtest.metrics import walk_forward_windows
from quant.common.logger import logger


@dataclass
class WalkForwardSummary:
    n_windows: int
    sharpe_mean: float
    sharpe_std: float
    sharpe_min: float
    sharpe_max: float
    cagr_mean: float
    cagr_std: float
    mdd_mean: float
    mdd_worst: float
    win_window_pct: float  # CAGR>0 윈도우 비율
    per_window: pd.DataFrame  # 윈도우별 상세

    def report(self) -> str:
        return (
            f"Walk-forward ({self.n_windows} windows)\n"
            f"  Sharpe (OOS) : {self.sharpe_mean:+.2f} ± {self.sharpe_std:.2f} "
            f"[min {self.sharpe_min:+.2f}, max {self.sharpe_max:+.2f}]\n"
            f"  CAGR   (OOS) : {self.cagr_mean * 100:+.2f}% ± {self.cagr_std * 100:.2f}%\n"
            f"  MDD    (mean): {self.mdd_mean * 100:.2f}%  /  worst: {self.mdd_worst * 100:.2f}%\n"
            f"  Win 윈도우   : {self.win_window_pct * 100:.1f}% (CAGR>0인 윈도우 비율)\n"
        )


def run_walk_forward(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    train_years: int = 1,
    test_months: int = 6,
    step_months: int = 6,
    cost_model: CostModel = BLUECHIP_KIS,
) -> WalkForwardSummary:
    """rolling walk-forward 백테스트 실행.

    train_years/test_months는 메타데이터 (시그널 자체가 lookback을 포함하므로
    실제 학습은 데이터 시작부터 됨). 본 함수는 OOS 구간 분포 평가에 집중.

    윈도우 정의:
      [start, start+train_years]  — 시그널 워밍업 + (가상의) 학습
      [start+train_years, +test_months]  — OOS 평가 구간
    """
    start = prices.index[0].date()
    end = prices.index[-1].date()
    windows = walk_forward_windows(
        start=start,
        end=end,
        train_years=train_years,
        test_months=test_months,
        step_months=step_months,
    )
    if not windows:
        raise ValueError("walk-forward 윈도우가 0개 — 데이터 기간 부족")

    rows = []
    for i, w in enumerate(windows, 1):
        oos_idx = (prices.index.date >= w.test_start) & (prices.index.date <= w.test_end)
        oos_dates = prices.index[oos_idx]
        if len(oos_dates) < 5:
            continue
        res = run_backtest(
            prices.loc[oos_dates],
            weights.loc[oos_dates],
            cost_model=cost_model,
        )
        m = res.metrics
        rows.append(
            {
                "window": i,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "days": m.days,
                "cagr": m.cagr,
                "sharpe": m.sharpe,
                "mdd": m.max_drawdown,
                "n_trades": m.n_trades,
                "turnover": m.turnover_annual or 0.0,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("walk-forward 결과 비어있음")

    summary = WalkForwardSummary(
        n_windows=len(df),
        sharpe_mean=float(df["sharpe"].mean()),
        sharpe_std=float(df["sharpe"].std(ddof=0)),
        sharpe_min=float(df["sharpe"].min()),
        sharpe_max=float(df["sharpe"].max()),
        cagr_mean=float(df["cagr"].mean()),
        cagr_std=float(df["cagr"].std(ddof=0)),
        mdd_mean=float(df["mdd"].mean()),
        mdd_worst=float(df["mdd"].min()),
        win_window_pct=float((df["cagr"] > 0).mean()),
        per_window=df,
    )
    logger.info(summary.report())

    # 안정성 가드레일
    if summary.sharpe_std > abs(summary.sharpe_mean) * 1.5 and summary.n_windows >= 3:
        logger.warning("⚠️ Sharpe 변동성이 평균보다 큼 — 윈도우 의존성 강함, 알파 안정성 의심.")
    if summary.win_window_pct < 0.5:
        logger.warning(
            f"⚠️ Win 윈도우 비율 {summary.win_window_pct * 100:.0f}% < 50% — 운 의존 가능성."
        )

    return summary


def _ignore_unused_imports() -> None:
    _ = (date, timedelta, np)
