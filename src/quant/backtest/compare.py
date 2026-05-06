"""전략 비교 리포트.

같은 데이터·기간·비용으로 여러 전략을 돌려 동일 척도로 비교.
- 모멘텀 Top-N
- 평균회귀
- 50/50 결합 (분산 효과 확인)
- 벤치마크 (Equal-weighted KOSPI 200)

가드레일 §1: 단순 베이스라인을 못 이기면 의미 없다.
가드레일 §5: 단일 전략보다 결합 시 Sharpe·MDD 모두 좋아질 수 있는지.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from quant.backtest.benchmark import equal_weight_universe
from quant.backtest.costs import BLUECHIP_KIS, CostModel
from quant.backtest.engine import BacktestResult, run_backtest
from quant.common.logger import logger
from quant.strategies import mean_reversion, momentum_topn


def run_compare(
    prices: pd.DataFrame,
    *,
    values: pd.DataFrame | None = None,
    momentum_cfg: momentum_topn.MomentumTopNConfig | None = None,
    meanrev_cfg: mean_reversion.MeanReversionConfig | None = None,
    cost_model: CostModel = BLUECHIP_KIS,
    out_dir: Path | None = None,
) -> dict[str, BacktestResult]:
    """모든 전략 백테스트 + 비교 plot 생성.

    Returns: 전략명 → BacktestResult
    """
    momentum_cfg = momentum_cfg or momentum_topn.MomentumTopNConfig()
    meanrev_cfg = meanrev_cfg or mean_reversion.MeanReversionConfig()

    logger.info("[1/4] 모멘텀 Top-N 백테스트...")
    w_mom = momentum_topn.generate_weights(prices, values=values, config=momentum_cfg)
    res_mom = run_backtest(prices, w_mom, cost_model=cost_model)

    logger.info("[2/4] 평균회귀 백테스트...")
    w_mr = mean_reversion.generate_weights(prices, values=values, config=meanrev_cfg)
    res_mr = run_backtest(prices, w_mr, cost_model=cost_model)

    logger.info("[3/4] 50/50 결합 백테스트...")
    # 두 전략 가중치를 평균 → 합계 1.0 미만일 수 있음 (둘 다 0인 종목 많을 때)
    # 평균을 다시 합계 1.0으로 정규화하지 않음 (현금 보유 효과 허용)
    w_combo = (w_mom + w_mr) / 2.0
    res_combo = run_backtest(prices, w_combo, cost_model=cost_model)

    logger.info("[4/4] 벤치마크 (Equal-weighted KOSPI 200) 백테스트...")
    w_bench = equal_weight_universe(prices)
    res_bench = run_backtest(prices, w_bench, cost_model=cost_model)

    results: dict[str, BacktestResult] = {
        "momentum": res_mom,
        "mean_reversion": res_mr,
        "combined": res_combo,
        "benchmark": res_bench,
    }

    _print_table(results)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_compare(out_dir, results)
        logger.info(f"비교 결과 저장: {out_dir}")

    return results


def _print_table(results: dict[str, BacktestResult]) -> None:
    """전략별 핵심 지표 표."""
    bar = "=" * 80
    logger.info(f"\n{bar}\n전략 비교\n{bar}")
    header = f"{'전략':16s} {'CAGR':>8s} {'Sharpe':>7s} {'MDD':>8s} {'Calmar':>7s} {'Trades':>7s} {'Turn/Y':>7s}"
    logger.info(header)
    logger.info("-" * 80)
    for name, res in results.items():
        m = res.metrics
        logger.info(
            f"{name:16s} {m.cagr * 100:>7.2f}% "
            f"{m.sharpe:>+7.2f} "
            f"{m.max_drawdown * 100:>+7.2f}% "
            f"{m.calmar:>7.2f} "
            f"{m.n_trades:>7d} "
            f"{(m.turnover_annual or 0):>7.1f}"
        )
    logger.info(bar)

    # 가드레일 §1: 베이스라인을 이기는가?
    bench = results.get("benchmark")
    if bench is not None:
        bench_sharpe = bench.metrics.sharpe
        for name, res in results.items():
            if name == "benchmark":
                continue
            edge = res.metrics.sharpe - bench_sharpe
            if edge > 0:
                logger.info(f"  ✅ {name} Sharpe edge vs benchmark: +{edge:.2f}")
            else:
                logger.warning(
                    f"  ⚠️ {name} Sharpe edge vs benchmark: {edge:+.2f} (베이스라인 못 이김)"
                )


def _save_compare(out_dir: Path, results: dict[str, BacktestResult]) -> None:
    # CSV — 일별 자본 곡선 한 파일에 모음
    eq_df = pd.DataFrame({name: r.equity_curve for name, r in results.items()})
    eq_df.to_csv(out_dir / "equity_curves.csv")

    # 메트릭 JSON
    summary = {
        name: {
            "cagr": r.metrics.cagr,
            "sharpe": r.metrics.sharpe,
            "sortino": r.metrics.sortino,
            "mdd": r.metrics.max_drawdown,
            "calmar": r.metrics.calmar,
            "n_trades": r.metrics.n_trades,
            "turnover_annual": r.metrics.turnover_annual,
            "win_rate": r.metrics.win_rate,
            "warnings": r.metrics.warnings,
        }
        for name, r in results.items()
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Plot — equity overlay
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, r in results.items():
        ax.plot(r.equity_curve.index, r.equity_curve.values, label=name, linewidth=1.3)
    ax.set_title("Strategy Comparison — Equity Curves")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_equity.png", dpi=120)
    plt.close(fig)

    # Drawdown overlay
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, r in results.items():
        eq = r.equity_curve
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=name, linewidth=1.0)
    ax.set_title("Strategy Comparison — Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_drawdown.png", dpi=120)
    plt.close(fig)
