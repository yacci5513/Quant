"""fetch_krx 유틸 검증 (네트워크 미사용 부분만)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant.data.price.fetch_krx import (
    _last_stored_date,
    _ticker_path,
    load_close_panel,
    save_ohlcv,
)


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return tmp_path / "prices"


def _sample_df(start: str, n: int) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq="B", name="date")
    return pd.DataFrame(
        {
            "open": range(n),
            "high": range(n),
            "low": range(n),
            "close": [100 + i for i in range(n)],
            "volume": [1000] * n,
            "value": [100_000] * n,
            "change_pct": [0.0] * n,
        },
        index=idx,
    )


def test_save_then_read_roundtrip(base: Path) -> None:
    df = _sample_df("2025-01-02", 5)
    path = save_ohlcv("005930", df, base)
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == 5
    assert list(loaded.columns) == list(df.columns)


def test_save_merges_and_dedupes(base: Path) -> None:
    """겹치는 일자가 있으면 새로운 데이터로 덮어쓴다."""
    first = _sample_df("2025-01-02", 5)  # 1/2, 1/3, 1/6, 1/7, 1/8
    save_ohlcv("005930", first, base)
    second = _sample_df("2025-01-08", 5)  # 1/8, 1/9, 1/10, 1/13, 1/14 — 1/8 겹침
    save_ohlcv("005930", second, base)
    loaded = pd.read_parquet(_ticker_path("005930", base))
    assert len(loaded) == 9  # 5 + 5 - 1(겹침) = 9
    assert loaded.index.is_monotonic_increasing
    assert not loaded.index.has_duplicates


def test_last_stored_date_none_when_missing(base: Path) -> None:
    assert _last_stored_date(_ticker_path("999999", base)) is None


def test_last_stored_date_returns_max(base: Path) -> None:
    df = _sample_df("2025-01-02", 3)
    save_ohlcv("005930", df, base)
    last = _last_stored_date(_ticker_path("005930", base))
    assert last == date(2025, 1, 6)  # 1/2 월, 1/3 화, 1/6 월(영업일)


def test_load_close_panel(base: Path) -> None:
    save_ohlcv("005930", _sample_df("2025-01-02", 3), base)
    save_ohlcv("000660", _sample_df("2025-01-02", 3), base)
    panel = load_close_panel(base=base)
    assert set(panel.columns) == {"005930", "000660"}
    assert len(panel) == 3
