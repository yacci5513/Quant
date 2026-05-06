"""펀더멘털 시그널 모듈 테스트 (네트워크 미사용)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.fundamental import (
    FundamentalConfig,
    _financials_to_daily_panel,
    _zscore_cross_sectional,
    generate_weights,
)


def test_zscore_basic() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = _zscore_cross_sectional(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


def test_zscore_handles_few_valid() -> None:
    s = pd.Series([1.0, np.nan, np.nan, np.nan])
    z = _zscore_cross_sectional(s)
    # 유효 < 5개면 모두 0 (중립)
    assert (z == 0).all()


def test_zscore_fills_nan_with_zero() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan])
    z = _zscore_cross_sectional(s)
    # NaN 위치는 0
    assert z.iloc[5] == 0.0
    assert z.iloc[6] == 0.0


def test_financials_to_daily_panel_applies_lag() -> None:
    """bsns_year=2023 사업보고서는 다음 해(2024) + lag일 후부터 적용."""
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    prices = pd.DataFrame(100, index=idx, columns=["A", "B"], dtype=float)
    financials = pd.DataFrame(
        [
            {
                "stock_code": "A",
                "bsns_year": 2023,
                "net_income": 1000,
                "equity": 5000,
                "revenue": 10000,
                "operating_income": 800,
                "assets": 8000,
            },
            {
                "stock_code": "B",
                "bsns_year": 2023,
                "net_income": 2000,
                "equity": 8000,
                "revenue": 20000,
                "operating_income": 1500,
                "assets": 12000,
            },
        ]
    )
    panels = _financials_to_daily_panel(financials, prices, publication_lag_days=10)
    # net_income 패널 — 처음 10일은 NaN, 이후엔 2023 값
    assert pd.isna(panels["net_income"].iloc[0]["A"])
    assert panels["net_income"].iloc[15]["A"] == 1000
    assert panels["equity"].iloc[15]["B"] == 8000


def test_generate_weights_returns_top_n() -> None:
    """단순 재무 데이터로 weight 생성. ROE 큰 종목 선호."""
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    cols = ["A", "B", "C", "D", "E", "F", "G"]
    prices = pd.DataFrame(100, index=idx, columns=cols, dtype=float)
    values = prices * 1e7  # 거래대금 충분

    # A는 ROE 매우 높음, B 중간, C-G 낮음 (적자 포함)
    rows = []
    for i, c in enumerate(cols):
        ne = 1000 - i * 200  # A=1000, B=800, ..., F=0, G=-200 (적자)
        rows.append(
            {
                "stock_code": c,
                "bsns_year": 2023,
                "net_income": ne,
                "equity": 5000,
                "revenue": 10000,
                "operating_income": ne,
                "assets": 8000,
            }
        )
    financials = pd.DataFrame(rows)
    shares = pd.Series({c: 1_000_000 for c in cols}, dtype=float)

    weights = generate_weights(
        prices,
        financials,
        shares=shares,
        values=values,
        config=FundamentalConfig(
            top_n=3, rebalance_freq="BMS", publication_lag_days=10, min_avg_value=0
        ),
    )
    # ROE 가장 큰 A, B, C가 보유 종목 후보 (적자 D 이후는 제외)
    last = weights.iloc[-1]
    held = set(last[last > 0].index)
    assert len(held) == 3
    # F, G는 적자로 제외 가능성 높음
    assert "G" not in held


def test_generate_weights_empty_financials() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    prices = pd.DataFrame(100, index=idx, columns=["A"], dtype=float)
    shares = pd.Series({"A": 1_000_000})
    weights = generate_weights(prices, pd.DataFrame(), shares=shares)
    assert weights.empty or (weights == 0).all().all()


def test_generate_weights_skips_when_insufficient() -> None:
    """유효 종목 < top_n이면 그 리밸런싱 스킵."""
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    cols = ["A", "B", "C"]
    prices = pd.DataFrame(100, index=idx, columns=cols, dtype=float)
    values = prices * 1e7
    rows = [
        {
            "stock_code": c,
            "bsns_year": 2023,
            "net_income": 1000,
            "equity": 5000,
            "revenue": 10000,
            "operating_income": 800,
            "assets": 8000,
        }
        for c in cols
    ]
    financials = pd.DataFrame(rows)
    shares = pd.Series({c: 1_000_000 for c in cols}, dtype=float)
    # top_n=10 요구 — 3종목만 있어 모든 리밸런싱 스킵
    weights = generate_weights(
        prices,
        financials,
        shares=shares,
        values=values,
        config=FundamentalConfig(top_n=10, publication_lag_days=10, min_avg_value=0),
    )
    assert (weights.sum(axis=1) == 0).all()
