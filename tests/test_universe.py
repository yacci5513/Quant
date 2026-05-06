"""시점별 종목 유니버스 테스트 (네트워크 미사용)."""

from __future__ import annotations

import pandas as pd

from quant.data.price.universe import (
    market_cap_panel,
    time_series_universe_mask,
    universe_at,
)


def _synthetic() -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    cols = ["A", "B", "C", "D", "E"]
    # A: 가장 비싼 주식, E: 가장 싼 주식
    base = {"A": 100_000, "B": 50_000, "C": 30_000, "D": 20_000, "E": 10_000}
    prices = pd.DataFrame({c: base[c] for c in cols}, index=idx, dtype=float)
    # 미세한 변동 추가
    prices = prices * (
        1
        + 0.0001 * pd.DataFrame([[i, i, i, i, i] for i in range(len(idx))], index=idx, columns=cols)
    )
    # 발행주식수: A는 적게, E는 많게 → 실제론 A가 시총 큰 경우와 작은 경우 둘다 가능
    shares = pd.Series(
        {"A": 1_000_000, "B": 5_000_000, "C": 10_000_000, "D": 20_000_000, "E": 50_000_000}
    )
    return prices, shares


def test_market_cap_panel_shape() -> None:
    prices, shares = _synthetic()
    cap = market_cap_panel(prices, shares=shares)
    assert cap.shape == prices.shape
    # 첫 일자의 A 시가총액 = 100,000 × 1,000,000
    first = cap.iloc[0]
    assert abs(first["A"] - 100_000 * 1_000_000) < 100


def test_universe_at_top_n() -> None:
    prices, shares = _synthetic()
    cap = market_cap_panel(prices, shares=shares)
    # 합산 시총: A=1e11, B=2.5e11, C=3e11, D=4e11, E=5e11
    # 즉 E > D > C > B > A
    top_3 = universe_at(cap, prices.index[10], top_n=3)
    assert top_3 == ["E", "D", "C"]


def test_universe_at_falls_back_to_prior_date() -> None:
    prices, shares = _synthetic()
    cap = market_cap_panel(prices, shares=shares)
    future = prices.index[-1] + pd.Timedelta(days=10)
    # 미래 일자 → 직전 영업일 결과
    top_2 = universe_at(cap, future, top_n=2)
    assert len(top_2) == 2


def test_time_series_mask_propagates_forward() -> None:
    prices, shares = _synthetic()
    cap = market_cap_panel(prices, shares=shares)
    rb = prices.index[[0, 30, 60]]
    mask = time_series_universe_mask(cap, rb, top_n=3)
    # 3종목만 True
    assert (mask.sum(axis=1) == 3).all()


def test_market_cap_panel_filters_unknown_tickers() -> None:
    prices, shares = _synthetic()
    # shares 인덱스에서 B 제거
    shares_partial = shares.drop("B")
    cap = market_cap_panel(prices, shares=shares_partial)
    assert "B" not in cap.columns
    assert "A" in cap.columns
