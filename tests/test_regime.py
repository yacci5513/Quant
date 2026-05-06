"""시장 레짐 필터 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.regime import (
    RegimeMode,
    apply_regime_filter,
    filter_weights,
    market_proxy,
    regime_exposure,
    regime_state,
)


def _trending_market(n: int = 400, slope: float = 0.001) -> pd.DataFrame:
    """완만한 상승 후 절반에서 급락하는 합성 가격 패널."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    # 전반: 완만한 상승, 후반: 급락
    rets = np.concatenate(
        [
            rng.normal(slope, 0.005, n // 2),
            rng.normal(-0.003, 0.010, n - n // 2),
        ]
    )
    base = 1000 * (1 + pd.Series(rets, index=idx)).cumprod()
    return pd.DataFrame({f"T{i}": base * (1 + 0.001 * i) for i in range(5)}, index=idx)


def test_market_proxy_is_average() -> None:
    df = pd.DataFrame(
        {"A": [10.0, 12.0], "B": [20.0, 18.0]}, index=pd.date_range("2024-01-01", periods=2)
    )
    proxy = market_proxy(df)
    assert proxy.iloc[0] == 15.0
    assert proxy.iloc[1] == 15.0


def test_regime_exposure_binary_uptrend() -> None:
    # 단조 증가 → MA보다 항상 위
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    series = pd.Series(np.linspace(100, 200, 300), index=idx)
    exposure = regime_exposure(series, ma_window=50, mode=RegimeMode.BINARY)
    # 워밍업 후 모두 1.0
    warm = exposure.iloc[60:]
    assert (warm == 1.0).all()


def test_regime_exposure_binary_downtrend() -> None:
    # 단조 감소 → MA보다 항상 아래
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    series = pd.Series(np.linspace(200, 100, 300), index=idx)
    exposure = regime_exposure(series, ma_window=50, mode=RegimeMode.BINARY)
    warm = exposure.iloc[60:]
    assert (warm == 0.0).all()


def test_regime_exposure_half_mode() -> None:
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    series = pd.Series(np.linspace(200, 100, 300), index=idx)
    exposure = regime_exposure(series, ma_window=50, mode=RegimeMode.HALF)
    warm = exposure.iloc[60:]
    # 하락장이지만 절반은 유지
    assert (warm == 0.5).all()


def test_regime_off_returns_ones() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    series = pd.Series(np.linspace(200, 100, 100), index=idx)
    exposure = regime_exposure(series, ma_window=50, mode=RegimeMode.OFF)
    assert (exposure == 1.0).all()


def test_apply_regime_filter_zero_weights_in_downtrend() -> None:
    prices = _trending_market(n=400)
    proxy = market_proxy(prices)
    exposure = regime_exposure(proxy, ma_window=100, mode=RegimeMode.BINARY)
    weights = pd.DataFrame(0.2, index=prices.index, columns=prices.columns)
    filtered = apply_regime_filter(weights, exposure)
    # 후반(하락장)엔 가중치 0이어야 함 (이미 워밍업 끝난 후)
    tail = filtered.iloc[-50:]
    assert (tail.sum(axis=1) == 0).any()  # 적어도 일부 일자는 청산
    # 전반(상승장)엔 가중치 유지
    head_post_warm = filtered.iloc[150:200]
    assert (head_post_warm.sum(axis=1) > 0).any()


def test_filter_weights_returns_exposure() -> None:
    prices = _trending_market(n=300)
    weights = pd.DataFrame(0.1, index=prices.index, columns=prices.columns)
    filtered, exposure = filter_weights(prices, weights, ma_window=100)
    assert isinstance(exposure, pd.Series)
    assert filtered.shape == weights.shape


def test_regime_state_uptrend() -> None:
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    df = pd.DataFrame(
        {f"T{i}": np.linspace(100, 200, 300) for i in range(3)},
        index=idx,
    )
    state = regime_state(df, ma_window=50)
    assert state["in_uptrend"]
    assert state["distance_pct"] > 0
    assert "풀투자" in state["recommendation"]


def test_regime_state_downtrend() -> None:
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    df = pd.DataFrame(
        {f"T{i}": np.linspace(200, 100, 300) for i in range(3)},
        index=idx,
    )
    state = regime_state(df, ma_window=50)
    assert not state["in_uptrend"]
    assert state["distance_pct"] < 0
    assert "현금화" in state["recommendation"]
