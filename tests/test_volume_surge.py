"""거래량 급증 전략 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.volume_surge import VolumeSurgeConfig, generate_weights


def _synthetic_with_surge() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """전반부엔 일정 거래량, 후반부엔 일부 종목 급증 + 가격 상승."""
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    cols = [f"T{i}" for i in range(10)]

    # 가격: 후반부 5종목은 상승, 5종목은 보합
    base = []
    for i, _c in enumerate(cols):
        rets = np.zeros(200)
        rets[:120] = rng.normal(0, 0.005, 120)
        if i < 5:
            rets[120:] = rng.normal(0.005, 0.01, 80)  # 후반 상승
        else:
            rets[120:] = rng.normal(0, 0.005, 80)
        base.append(100 * (1 + pd.Series(rets, index=idx)).cumprod())
    prices = pd.concat(base, axis=1)
    prices.columns = cols

    # 거래량: 같은 종목들이 후반에 급증 (3배 이상)
    vols = pd.DataFrame(1_000_000, index=idx, columns=cols, dtype=float)
    vols.iloc[120:, :5] = vols.iloc[120:, :5] * 4  # 4배 급증
    values = prices * vols
    return prices, vols, values


def test_volume_surge_picks_surging_tickers() -> None:
    prices, vols, values = _synthetic_with_surge()
    weights = generate_weights(
        prices,
        volumes=vols,
        values=values,
        config=VolumeSurgeConfig(
            short_window=5,
            long_window=60,
            ratio_threshold=2.0,
            top_n=3,
            rebalance_freq="BMS",
            min_avg_value=0,
        ),
    )
    last = weights.iloc[-1]
    held = set(last[last > 0].index)
    # 급증한 0~4만 보유 가능
    assert held.issubset({"T0", "T1", "T2", "T3", "T4"})


def test_volume_surge_no_surge_means_empty() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    cols = [f"T{i}" for i in range(8)]
    prices = pd.DataFrame(100, index=idx, columns=cols, dtype=float)
    vols = pd.DataFrame(1_000_000, index=idx, columns=cols, dtype=float)
    values = prices * vols
    weights = generate_weights(
        prices,
        volumes=vols,
        values=values,
        config=VolumeSurgeConfig(top_n=3, ratio_threshold=2.0, min_avg_value=0),
    )
    # 거래량 급증 없음 + 가격 상승 없음 → 모든 가중치 0
    assert (weights.iloc[-30:].sum(axis=1) == 0).all()


def test_volume_surge_weights_sum_to_one_when_held() -> None:
    prices, vols, values = _synthetic_with_surge()
    weights = generate_weights(
        prices,
        volumes=vols,
        values=values,
        config=VolumeSurgeConfig(
            top_n=3, ratio_threshold=2.0, min_avg_value=0, rebalance_freq="BMS"
        ),
    )
    rb_sums = weights.sum(axis=1)
    valid = rb_sums[rb_sums > 0]
    assert (abs(valid - 1.0) < 1e-6).all()


def test_volume_surge_no_volumes_returns_empty() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = pd.DataFrame(100, index=idx, columns=["A", "B"], dtype=float)
    weights = generate_weights(prices, volumes=None, config=VolumeSurgeConfig())
    assert weights.empty
