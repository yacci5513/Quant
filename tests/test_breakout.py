"""52주 신고가 돌파 전략 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.breakout import BreakoutConfig, generate_weights


def _synthetic_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    cols = [f"T{i}" for i in range(15)]
    base = pd.DataFrame(
        {c: 100 * (1 + rng.normal(0.001 * i, 0.01, 400)).cumprod() for i, c in enumerate(cols)},
        index=idx,
    )
    values = base * 1e6  # 거래대금 풍부
    return base, values


def test_breakout_returns_top_n_weights() -> None:
    prices, values = _synthetic_panel()
    weights = generate_weights(
        prices,
        values=values,
        config=BreakoutConfig(window_days=100, top_n=5, rebalance_freq="BMS", proximity=0.95),
    )
    # rebalance 일자에 weights 합이 1.0 또는 0.0
    rb_sums = weights.sum(axis=1)
    valid_sums = rb_sums[rb_sums > 0]
    assert (abs(valid_sums - 1.0) < 1e-6).all()


def test_breakout_top_n_count() -> None:
    prices, values = _synthetic_panel()
    weights = generate_weights(
        prices,
        values=values,
        config=BreakoutConfig(window_days=100, top_n=5, rebalance_freq="BMS", proximity=0.0),
    )
    # 마지막 일자에 5종목 보유
    last = weights.iloc[-1]
    nonzero = last[last > 0]
    assert len(nonzero) == 5


def test_breakout_skips_when_no_breakout() -> None:
    # 모든 종목이 하락 추세 → 신고가 근처 종목 거의 없음
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    prices = pd.DataFrame(
        {f"T{i}": np.linspace(100, 50, 300) for i in range(10)},
        index=idx,
    )
    values = prices * 1e6
    weights = generate_weights(
        prices,
        values=values,
        config=BreakoutConfig(window_days=100, top_n=5, proximity=0.99),
    )
    # 후반부 가중치는 조건 미충족으로 0
    tail_sum = weights.iloc[-30:].sum(axis=1)
    assert (tail_sum == 0).any()


def test_breakout_empty_panel_returns_empty() -> None:
    empty = pd.DataFrame()
    weights = generate_weights(empty, config=BreakoutConfig())
    assert weights.empty
