"""평균회귀 + 변동성조정 모멘텀 + 거래이력 단위 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.audit import build_rebalance_log, summarize_log
from quant.backtest.benchmark import equal_weight_universe
from quant.backtest.engine import run_backtest
from quant.strategies.mean_reversion import MeanReversionConfig
from quant.strategies.mean_reversion import generate_weights as mr_w
from quant.strategies.momentum_volscaled import MomentumVolScaledConfig
from quant.strategies.momentum_volscaled import generate_weights as vs_w


def _synthetic_panel(n_days: int = 600, n_tickers: int = 20, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rets = rng.normal(0.0005, 0.015, (n_days, n_tickers))
    rets[0] = 0
    cols = [f"T{i}" for i in range(n_tickers)]
    return 100 * pd.DataFrame(rets, index=idx, columns=cols).add(1).cumprod()


# -----------------------------------------------------------------------------
# benchmark
# -----------------------------------------------------------------------------
def test_benchmark_weights_sum_to_one() -> None:
    prices = _synthetic_panel()
    w = equal_weight_universe(prices)
    sums = w.sum(axis=1).round(6)
    # 모든 가격이 유효하면 합 = 1.0
    assert ((sums == 1.0) | (sums == 0.0)).all()


# -----------------------------------------------------------------------------
# mean reversion
# -----------------------------------------------------------------------------
def test_mean_reversion_weights_valid() -> None:
    prices = _synthetic_panel(n_days=400, n_tickers=20)
    cfg = MeanReversionConfig(
        top_n=5, short_lookback_days=5, long_lookback_months=6, min_avg_value=0
    )
    w = mr_w(prices, values=None, config=cfg)
    sums = w.sum(axis=1).round(6)
    assert (sums <= 1.0001).all()
    assert (sums >= 0).all()


# -----------------------------------------------------------------------------
# volatility scaled
# -----------------------------------------------------------------------------
def test_volscaled_weights_normalized() -> None:
    prices = _synthetic_panel(n_days=500, n_tickers=20)
    cfg = MomentumVolScaledConfig(top_n=5, lookback_months=6, vol_window_days=30, min_avg_value=0)
    w = vs_w(prices, values=None, config=cfg)
    # 활성 행에선 합 = 1.0 (또는 0)
    sums = w.sum(axis=1).round(6)
    nonzero = sums[sums > 0]
    if len(nonzero) > 0:
        assert (abs(nonzero - 1.0) < 1e-6).all()


def test_volscaled_low_vol_gets_higher_weight() -> None:
    """변동성 작은 종목이 더 큰 가중치를 받아야 한다."""
    n_days = 500
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(1)
    # T0~T4: 저변동, T5~T9: 고변동, 모두 같은 드리프트
    sigs = np.array([0.005] * 5 + [0.02] * 5)
    rets = rng.normal(0.001, sigs, (n_days, 10))
    rets[0] = 0
    prices = (
        100 * pd.DataFrame(rets, index=idx, columns=[f"T{i}" for i in range(10)]).add(1).cumprod()
    )

    cfg = MomentumVolScaledConfig(
        top_n=10, lookback_months=3, skip_months=0, vol_window_days=30, min_avg_value=0
    )
    w = vs_w(prices, values=None, config=cfg)
    last_w = w[w.sum(axis=1) > 0].iloc[-1]
    low_vol_total = last_w.iloc[:5].sum()
    high_vol_total = last_w.iloc[5:].sum()
    assert (
        low_vol_total > high_vol_total
    ), f"저변동 합 {low_vol_total:.3f} > 고변동 합 {high_vol_total:.3f} 기대"


# -----------------------------------------------------------------------------
# audit
# -----------------------------------------------------------------------------
def test_audit_log_only_at_rebalance_dates() -> None:
    """forward-fill 사이 날짜는 로그에 안 들어가야 한다."""
    prices = _synthetic_panel(n_days=200, n_tickers=10)
    rb1 = prices.index[20]
    rb2 = prices.index[100]

    # 명시적으로 rb1: T0만, rb2: T1만 (T0 청산) 가중치 구성
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    # rb1 ~ rb2-1: T0=1.0
    mask1 = (prices.index >= rb1) & (prices.index < rb2)
    weights.loc[mask1, "T0"] = 1.0
    # rb2 ~ end: T1=1.0
    mask2 = prices.index >= rb2
    weights.loc[mask2, "T1"] = 1.0

    log = build_rebalance_log(prices, weights)
    # 리밸런싱 시점 2개 × 보유 1종목 = 2행
    assert len(log) == 2
    assert log.iloc[0]["ticker"] == "T0"
    assert log.iloc[1]["ticker"] == "T1"


def test_audit_summary_aggregates() -> None:
    prices = _synthetic_panel(n_days=200, n_tickers=5)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[prices.index[10], "T0"] = 1.0
    weights.loc[prices.index[60], "T0"] = 1.0
    weights.loc[prices.index[110], "T1"] = 1.0
    weights = weights.replace(0, np.nan).ffill().fillna(0)
    log = build_rebalance_log(prices, weights)
    summary = summarize_log(log)
    assert "T0" in summary.index
    assert summary.loc["T0", "appearances"] >= 1


# -----------------------------------------------------------------------------
# integration: full pipeline benchmark vs strategy
# -----------------------------------------------------------------------------
def test_benchmark_runs_without_error() -> None:
    prices = _synthetic_panel(n_days=400, n_tickers=10)
    w = equal_weight_universe(prices)
    res = run_backtest(prices, w)
    assert res.metrics.days > 0
