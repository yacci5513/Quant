"""백테스트 시각화 (matplotlib, headless 호환).

함수들은 모두 figure를 반환 → 호출자가 savefig로 PNG 저장.
Docker 환경 (DISPLAY 없음) 대응 위해 Agg backend 강제.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity(
    equity: pd.Series,
    title: str = "Equity Curve",
    benchmark: pd.Series | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, label="strategy", linewidth=1.4)
    if benchmark is not None:
        # 시작점 1.0 정규화
        bench = benchmark / benchmark.iloc[0]
        ax.plot(bench.index, bench.values, label="benchmark", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Equity (start = 1.0)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> plt.Figure:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.4)
    ax.plot(dd.index, dd.values, color="tab:red", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    ax.set_ylim(min(dd.min() * 1.1, -0.05), 0.01)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_monthly_heatmap(returns: pd.Series, title: str = "Monthly Returns") -> plt.Figure:
    """월별 수익률 히트맵 (year × month)."""
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    monthly.index = pd.MultiIndex.from_arrays([monthly.index.year, monthly.index.month])
    pivot = monthly.unstack(level=-1)
    pivot.columns = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]
        for m in pivot.columns
    ]

    fig, ax = plt.subplots(figsize=(11, max(3, len(pivot) * 0.3)))
    data = pivot.values
    cmap = plt.get_cmap("RdYlGn")
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="monthly return")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_sweep_heatmap(
    pivot: pd.DataFrame,
    title: str = "Parameter Sweep (Sharpe)",
    cmap: str = "viridis",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    data = pivot.values
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(pivot.columns.name or "lookback")
    ax.set_ylabel(pivot.index.name or "top_n")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def save_all(
    out_dir: Path,
    equity_is: pd.Series,
    equity_oos: pd.Series,
    returns_is: pd.Series,
    returns_oos: pd.Series,
) -> dict[str, Path]:
    """결과 묶음 저장. 반환값은 파일경로 매핑."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    f = plot_equity(equity_is, title="Equity (In-Sample)")
    p = out_dir / "equity_is.png"
    f.savefig(p, dpi=120)
    plt.close(f)
    paths["equity_is"] = p

    f = plot_equity(equity_oos, title="Equity (Out-of-Sample)")
    p = out_dir / "equity_oos.png"
    f.savefig(p, dpi=120)
    plt.close(f)
    paths["equity_oos"] = p

    f = plot_drawdown(equity_is, title="Drawdown (IS)")
    p = out_dir / "drawdown_is.png"
    f.savefig(p, dpi=120)
    plt.close(f)
    paths["drawdown_is"] = p

    f = plot_drawdown(equity_oos, title="Drawdown (OOS)")
    p = out_dir / "drawdown_oos.png"
    f.savefig(p, dpi=120)
    plt.close(f)
    paths["drawdown_oos"] = p

    full_returns = pd.concat([returns_is, returns_oos]).sort_index()
    f = plot_monthly_heatmap(full_returns, title="Monthly Returns (IS+OOS)")
    p = out_dir / "monthly_heatmap.png"
    f.savefig(p, dpi=120)
    plt.close(f)
    paths["monthly_heatmap"] = p

    return paths
