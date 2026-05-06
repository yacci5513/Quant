"""앙상블 (다중 시그널 결합) 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from quant.backtest.ensemble import StrategyAllocation, combine


@pytest.fixture
def simple_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    cols = ["A", "B", "C", "D"]
    # 전략1: A,B 보유 (50/50)
    s1 = pd.DataFrame(0.0, index=idx, columns=cols)
    s1["A"] = 0.5
    s1["B"] = 0.5
    # 전략2: C,D 보유 (50/50)
    s2 = pd.DataFrame(0.0, index=idx, columns=cols)
    s2["C"] = 0.5
    s2["D"] = 0.5
    return s1, s2


def test_combine_50_50_yields_uniform(simple_panels) -> None:
    s1, s2 = simple_panels
    out = combine(
        [
            StrategyAllocation("s1", s1, 0.5),
            StrategyAllocation("s2", s2, 0.5),
        ]
    )
    # 모든 종목 25% 동일가중이어야 함
    assert (abs(out - 0.25) < 1e-6).all().all()


def test_combine_60_40(simple_panels) -> None:
    s1, s2 = simple_panels
    out = combine(
        [
            StrategyAllocation("s1", s1, 0.6),
            StrategyAllocation("s2", s2, 0.4),
        ]
    )
    # A,B = 0.6 * 0.5 = 0.30,  C,D = 0.4 * 0.5 = 0.20
    last = out.iloc[-1]
    assert abs(last["A"] - 0.30) < 1e-6
    assert abs(last["B"] - 0.30) < 1e-6
    assert abs(last["C"] - 0.20) < 1e-6
    assert abs(last["D"] - 0.20) < 1e-6


def test_combine_normalizes_unequal_alloc(simple_panels, caplog) -> None:
    s1, s2 = simple_panels
    out = combine(
        [
            StrategyAllocation("s1", s1, 0.6),
            StrategyAllocation("s2", s2, 0.6),  # 합 1.2 → 정규화 필요
        ]
    )
    # 합 1.0 정규화 후 50/50 동일
    last = out.iloc[-1]
    assert abs(last.sum() - 1.0) < 1e-6


def test_combine_handles_disjoint_columns() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    s1 = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    s1["A"] = 1.0
    s2 = pd.DataFrame(0.0, index=idx, columns=["C", "D"])
    s2["C"] = 1.0
    out = combine(
        [
            StrategyAllocation("s1", s1, 0.5),
            StrategyAllocation("s2", s2, 0.5),
        ]
    )
    # 4개 컬럼 모두 등장
    assert set(out.columns) == {"A", "B", "C", "D"}
    last = out.iloc[-1]
    assert abs(last["A"] - 0.5) < 1e-6
    assert abs(last["C"] - 0.5) < 1e-6
    assert abs(last["B"]) < 1e-9
    assert abs(last["D"]) < 1e-9


def test_combine_empty_allocations_raises() -> None:
    with pytest.raises(ValueError):
        combine([])
