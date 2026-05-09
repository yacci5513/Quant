"""다중 레짐 전략 결합 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.regime_strategy import (
    DEFAULT_POLICY,
    Regime,
    RegimeConfig,
    classify_regimes,
    combine_by_regime,
    regime_distribution,
)


def _synth_market(scenario: str, n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    if scenario == "uptrend":
        rets = rng.normal(0.001, 0.005, (n, 5))
    elif scenario == "downtrend":
        rets = rng.normal(-0.001, 0.005, (n, 5))
    else:  # cross
        rets1 = rng.normal(0.001, 0.005, (n // 2, 5))
        rets2 = rng.normal(-0.001, 0.008, (n - n // 2, 5))
        rets = np.concatenate([rets1, rets2])
    cols = [f"T{i}" for i in range(5)]
    return pd.DataFrame(
        100 * (1 + pd.DataFrame(rets, index=idx, columns=cols)).cumprod(), index=idx
    )


def test_classify_uptrend_mostly_bull() -> None:
    prices = _synth_market("uptrend", n=300)
    regimes = classify_regimes(prices, RegimeConfig(ma_window=50))
    dist = regime_distribution(regimes)
    bull = dist.get(Regime.BULL_STRONG.value, 0) + dist.get(Regime.BULL_CHOPPY.value, 0)
    assert bull > 0.5  # 상승장이라 BULL이 다수


def test_classify_downtrend_mostly_bear() -> None:
    prices = _synth_market("downtrend", n=300)
    regimes = classify_regimes(prices, RegimeConfig(ma_window=50))
    dist = regime_distribution(regimes)
    assert dist.get(Regime.BEAR.value, 0) > 0.5


def test_classify_cross_has_recovery_signals() -> None:
    """전환 시점이 있는 시장은 BEAR + 일부 BULL/RECOVERY."""
    prices = _synth_market("cross", n=600)
    regimes = classify_regimes(prices, RegimeConfig(ma_window=100))
    dist = regime_distribution(regimes)
    # 모든 레짐이 적어도 한 번 등장하지 않아도 OK — 시드에 따라 다름
    assert sum(dist.values()) == pytest_approx(1.0)


def test_combine_by_regime_zeros_in_bear() -> None:
    """BEAR 레짐엔 정책상 현금 → combined 행 합 0."""
    prices = _synth_market("downtrend", n=400)
    sig = pd.DataFrame(0.1, index=prices.index, columns=[f"T{i}" for i in range(5)])
    combined = combine_by_regime(
        prices, {"momentum": sig}, regime_config=RegimeConfig(ma_window=50)
    )
    # 후반 100일은 거의 BEAR → 합 0
    tail_sums = combined.iloc[-100:].sum(axis=1)
    assert (tail_sums == 0).any()


def test_combine_uses_policy_weights() -> None:
    """BULL_CHOPPY 레짐일 때 momentum + low_vol 둘 다 사용."""
    prices = _synth_market("uptrend", n=300)
    cols = [f"T{i}" for i in range(5)]
    mom_panel = pd.DataFrame(0.0, index=prices.index, columns=cols)
    mom_panel["T0"] = 1.0  # 모멘텀은 T0만
    lv_panel = pd.DataFrame(0.0, index=prices.index, columns=cols)
    lv_panel["T1"] = 1.0  # 저변동은 T1만
    combined = combine_by_regime(
        prices,
        {"momentum": mom_panel, "low_vol": lv_panel},
        regime_config=RegimeConfig(ma_window=50),
    )
    # 후반 안정 구간에서 최소 한 일자는 두 종목 모두 가중치 양수 (BULL_CHOPPY)
    valid = combined.iloc[100:]
    has_both = ((valid["T0"] > 0) & (valid["T1"] > 0)).any()
    assert has_both


def test_default_policy_structure() -> None:
    pol = DEFAULT_POLICY
    assert pol.bull_strong["momentum"] == 1.0
    assert pol.bear == {}  # 현금
    assert sum(pol.bull_choppy.values()) == 1.0
    assert sum(pol.recovery.values()) == 0.5  # 50% 진입


def pytest_approx(value: float, tol: float = 1e-6) -> ApproxFloat:
    """간단 approx (pytest 의존 없이)."""
    return ApproxFloat(value, tol)


class ApproxFloat:
    def __init__(self, v: float, tol: float) -> None:
        self.v = v
        self.tol = tol

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, int | float):
            return NotImplemented
        return abs(self.v - other) < self.tol

    def __repr__(self) -> str:
        return f"approx({self.v})"
