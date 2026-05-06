"""자사주 매입 공시 모듈 테스트 (네트워크 미사용)."""

from __future__ import annotations

import pandas as pd

from quant.data.disclosure.treasury_stock import _classify, events_to_signal_panel


def test_classify_buyback_decision() -> None:
    assert _classify("주요사항보고서(자기주식취득결정)") == "buyback_decision"
    assert _classify("자사주취득결정") == "buyback_decision"


def test_classify_buyback_result() -> None:
    assert _classify("자기주식취득결과보고서") == "buyback_result"


def test_classify_cancellation() -> None:
    assert _classify("주식소각결정") == "cancellation"
    assert _classify("자기주식소각 결정") == "cancellation"


def test_classify_correction_excluded() -> None:
    """[기재정정] 접두 공시는 None — 첫 발견만 신호로 카운트."""
    assert _classify("[기재정정]주요사항보고서(자기주식취득결정)") is None


def test_classify_unrelated_returns_none() -> None:
    assert _classify("분기보고서") is None
    assert _classify("배당결정") is None


def test_signal_panel_assigns_decaying_score() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = pd.DataFrame(100, index=idx, columns=["A", "B", "C"], dtype=float)
    events = pd.DataFrame(
        [
            {
                "stock_code": "A",
                "corp_code": "00000001",
                "corp_name": "TestA",
                "rcept_dt": pd.Timestamp("2024-01-15"),
                "report_nm": "주식소각결정",
                "event_type": "cancellation",
                "rcept_no": "2024011500001",
            }
        ]
    )
    panel = events_to_signal_panel(events, prices, decay_days=10)
    # 공시 다음 영업일부터 양수
    after = panel.loc[panel.index > "2024-01-15", "A"]
    assert after.iloc[0] == 1.0  # cancellation 기본 1.0
    # 10영업일 후엔 0
    assert (after.iloc[10:] == 0).all()
    # 선형 감쇠
    assert after.iloc[0] > after.iloc[5] > after.iloc[9]


def test_signal_panel_max_overlap() -> None:
    """같은 종목 두 이벤트 중첩 시 max 점수 유지."""
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    prices = pd.DataFrame(100, index=idx, columns=["A"], dtype=float)
    events = pd.DataFrame(
        [
            {
                "stock_code": "A",
                "corp_code": "X",
                "corp_name": "T",
                "rcept_dt": pd.Timestamp("2024-01-15"),
                "report_nm": "buyback",
                "event_type": "buyback_decision",
                "rcept_no": "1",
            },
            {
                "stock_code": "A",
                "corp_code": "X",
                "corp_name": "T",
                "rcept_dt": pd.Timestamp("2024-01-20"),
                "report_nm": "cancel",
                "event_type": "cancellation",
                "rcept_no": "2",
            },
        ]
    )
    panel = events_to_signal_panel(events, prices, decay_days=10)
    # 1/22 시점엔 cancellation(1.0)이 buyback_decision(0.7)보다 우세
    after_cancel = panel.loc[panel.index > "2024-01-20", "A"].iloc[0]
    assert after_cancel == 1.0


def test_signal_panel_unknown_ticker_ignored() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    prices = pd.DataFrame(100, index=idx, columns=["A"], dtype=float)
    events = pd.DataFrame(
        [
            {
                "stock_code": "B",  # 패널에 없음
                "corp_code": "X",
                "corp_name": "T",
                "rcept_dt": pd.Timestamp("2024-01-02"),
                "report_nm": "X",
                "event_type": "cancellation",
                "rcept_no": "1",
            }
        ]
    )
    panel = events_to_signal_panel(events, prices)
    assert (panel == 0).all().all()


def test_signal_panel_empty_events_returns_zeros() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    prices = pd.DataFrame(100, index=idx, columns=["A", "B"], dtype=float)
    panel = events_to_signal_panel(pd.DataFrame(), prices)
    assert panel.shape == prices.shape
    assert (panel == 0).all().all()
