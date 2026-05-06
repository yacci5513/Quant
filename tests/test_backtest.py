"""백테스트 엔진 + 메트릭 + 전략 sanity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.costs import BLUECHIP_KIS, CONSERVATIVE
from quant.backtest.engine import run_backtest, run_split
from quant.backtest.metrics import (
    SUSPICIOUS_SHARPE,
    compute_metrics,
    split_in_out_of_sample,
    walk_forward_windows,
)
from quant.strategies.momentum_topn import MomentumTopNConfig, generate_weights


# -----------------------------------------------------------------------------
# costs
# -----------------------------------------------------------------------------
def test_bluechip_round_trip_within_range() -> None:
    """우량주 왕복 비용이 0.2~0.5% 범위에 있어야 한다 (가드레일 §4)."""
    rt = BLUECHIP_KIS.round_trip_cost
    assert 0.002 < rt < 0.005, f"unexpected round_trip={rt}"


def test_conservative_more_expensive_than_bluechip() -> None:
    assert CONSERVATIVE.round_trip_cost > BLUECHIP_KIS.round_trip_cost


# -----------------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------------
def test_metrics_zero_returns_yield_zero_sharpe() -> None:
    idx = pd.date_range("2024-01-01", periods=252, freq="B")
    rets = pd.Series(0.0, index=idx)
    m = compute_metrics(rets, n_trades=50)
    assert m.sharpe == 0.0
    assert m.total_return == 0.0


def test_metrics_warns_on_too_few_trades() -> None:
    idx = pd.date_range("2024-01-01", periods=252, freq="B")
    rets = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 252), index=idx)
    m = compute_metrics(rets, n_trades=5)
    assert any("거래 횟수" in w for w in m.warnings)


def test_metrics_warns_on_suspiciously_high_sharpe() -> None:
    idx = pd.date_range("2024-01-01", periods=252, freq="B")
    # 매일 +0.1% 일정 → 변동성 0 가까움 → Sharpe 폭발
    rets = pd.Series(0.001 + np.random.default_rng(0).normal(0, 1e-6, 252), index=idx)
    m = compute_metrics(rets, n_trades=100)
    assert m.sharpe > SUSPICIOUS_SHARPE
    assert any("과적합" in w for w in m.warnings)


def test_split_is_oos_correct_proportion() -> None:
    idx = pd.date_range("2020-01-01", periods=1000, freq="B")
    is_idx, oos_idx = split_in_out_of_sample(idx, is_ratio=0.7)
    assert len(is_idx) == 700
    assert len(oos_idx) == 300
    assert is_idx[-1] < oos_idx[0]


def test_walk_forward_windows_non_overlapping_test() -> None:
    from datetime import date

    windows = walk_forward_windows(
        start=date(2020, 1, 1),
        end=date(2025, 1, 1),
        train_years=1,
        test_months=6,
        step_months=6,
    )
    assert len(windows) > 0
    for i in range(1, len(windows)):
        # test 구간이 step만큼 슬라이딩
        assert windows[i].test_start > windows[i - 1].test_start


# -----------------------------------------------------------------------------
# engine
# -----------------------------------------------------------------------------
def _synthetic_panel(n_days: int = 250, n_tickers: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rets = rng.normal(0.0005, 0.015, (n_days, n_tickers))
    rets[0] = 0
    prices = (
        100
        * (1 + pd.DataFrame(rets, index=idx, columns=[f"T{i}" for i in range(n_tickers)])).cumprod()
    )
    return prices


def test_engine_buy_and_hold_matches_close_returns() -> None:
    """단일 종목 100% 비중 → 그 종목 수익률과 일치 (비용 제외)."""
    prices = _synthetic_panel(n_days=100, n_tickers=2)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights["T0"] = 1.0  # 항상 T0 100%
    res = run_backtest(prices, weights, cost_model=BLUECHIP_KIS)
    expected_total = prices["T0"].iloc[-1] / prices["T0"].iloc[0] - 1
    # 비용은 첫 진입에서만 발생 (이후 weights 변화 없음)
    cost_first_day = BLUECHIP_KIS.avg_fee_per_side  # turnover 1.0 한 번
    assert abs(res.metrics.total_return - (expected_total - cost_first_day)) < 0.005


def test_engine_validates_unsorted_index_raises() -> None:
    prices = _synthetic_panel()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    bad_prices = prices.iloc[::-1]  # 역순
    with pytest.raises(ValueError, match="정렬"):
        run_backtest(bad_prices, weights)


def test_engine_validates_no_common_columns_raises() -> None:
    prices = _synthetic_panel()
    weights = pd.DataFrame(
        0.0,
        index=prices.index,
        columns=["X", "Y"],  # 다른 종목명
    )
    with pytest.raises(ValueError, match="공통 종목"):
        run_backtest(prices, weights)


def test_engine_higher_cost_lowers_return() -> None:
    prices = _synthetic_panel(n_days=200)
    # 매일 가중치를 무작위로 바꿔 회전율 강제
    rng = np.random.default_rng(0)
    raw = rng.dirichlet(np.ones(prices.shape[1]), size=len(prices))
    weights = pd.DataFrame(raw, index=prices.index, columns=prices.columns)
    cheap = run_backtest(prices, weights, cost_model=BLUECHIP_KIS)
    expensive = run_backtest(prices, weights, cost_model=CONSERVATIVE)
    assert expensive.metrics.total_return < cheap.metrics.total_return


def test_run_split_returns_two_results() -> None:
    prices = _synthetic_panel(n_days=300)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights["T0"] = 1.0
    is_res, oos_res = run_split(prices, weights, is_ratio=0.7)
    assert is_res.metrics.days < oos_res.metrics.days * 3  # 대략 7:3
    assert is_res.metrics.period_end < oos_res.metrics.period_start


# -----------------------------------------------------------------------------
# momentum strategy
# -----------------------------------------------------------------------------
def test_momentum_picks_top_n() -> None:
    """단조 증가하는 가격일수록 모멘텀 점수 높음."""
    n_days = 400
    n_tickers = 20
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    # T0~T9: 강한 상승, T10~T19: 약한 상승
    rng = np.random.default_rng(42)
    drifts = np.concatenate(
        [
            np.full(10, 0.002),  # 강세
            np.full(10, 0.0001),  # 약세
        ]
    )
    rets = rng.normal(drifts, 0.01, (n_days, n_tickers))
    rets[0] = 0
    prices = (
        100
        * pd.DataFrame(rets, index=idx, columns=[f"T{i}" for i in range(n_tickers)])
        .add(1)
        .cumprod()
    )
    cfg = MomentumTopNConfig(top_n=5, lookback_months=6, skip_months=1, min_avg_value=0)
    w = generate_weights(prices, values=None, config=cfg)
    # 마지막 리밸런싱 시점에서 선정된 종목 — 강세 그룹에 더 몰려야 함
    last_rebal = w.loc[w.sum(axis=1) > 0].index[-1]
    selected = w.loc[last_rebal][w.loc[last_rebal] > 0].index.tolist()
    strong_picks = [s for s in selected if int(s[1:]) < 10]
    assert len(strong_picks) >= 3, f"강세 그룹 5개 중 3개 이상 선정 기대, 실제: {selected}"


def test_momentum_weights_sum_to_one_or_zero() -> None:
    prices = _synthetic_panel(n_days=400, n_tickers=20)
    cfg = MomentumTopNConfig(top_n=5, lookback_months=6, min_avg_value=0)
    w = generate_weights(prices, values=None, config=cfg)
    sums = w.sum(axis=1).round(6)
    # 0 또는 약 1.0
    assert ((sums == 0) | (abs(sums - 1.0) < 1e-6)).all()
