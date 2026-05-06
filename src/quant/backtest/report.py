"""quantstats HTML 리포트 생성.

전략 또는 비교 결과를 받아 풀 리포트(차트·통계 포함)를 HTML로 저장.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

# matplotlib 폰트 경고 등 묵음
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def generate_html(
    returns: pd.Series,
    output_path: Path,
    *,
    title: str = "Quant Backtest Report",
    benchmark: pd.Series | None = None,
) -> Path:
    """quantstats로 풀 리포트 생성.

    returns: 일별 수익률 (DatetimeIndex)
    benchmark: 비교 벤치마크 일별 수익률 (선택)
    """
    import quantstats as qs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(
        returns,
        benchmark=benchmark,
        output=str(output_path),
        title=title,
    )
    return output_path


def generate_basic_metrics(returns: pd.Series) -> dict[str, float]:
    """quantstats 핵심 통계만 dict로 반환 (HTML 없이)."""
    import quantstats as qs

    return {
        "cagr": float(qs.stats.cagr(returns)),
        "sharpe": float(qs.stats.sharpe(returns)),
        "sortino": float(qs.stats.sortino(returns)),
        "max_drawdown": float(qs.stats.max_drawdown(returns)),
        "calmar": float(qs.stats.calmar(returns)),
        "volatility": float(qs.stats.volatility(returns)),
        "skew": float(qs.stats.skew(returns)),
        "kurtosis": float(qs.stats.kurtosis(returns)),
        "var_95": float(qs.stats.var(returns)),
        "cvar_95": float(qs.stats.cvar(returns)),
    }
