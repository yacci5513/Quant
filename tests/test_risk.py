"""리스크 관리 모듈 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.risk import (
    ExposureLimits,
    KillSwitchConfig,
    check_exposure,
    evaluate_kill_switch,
    pre_trade_check,
    quarter_kelly_fraction,
)


# -----------------------------------------------------------------------------
# Kill switch
# -----------------------------------------------------------------------------
def test_kill_switch_normal_returns_no_trigger() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    rets = pd.Series(np.random.default_rng(0).normal(0.001, 0.005, 100), index=idx)
    state = evaluate_kill_switch(rets)
    assert not state.triggered


def test_kill_switch_triggers_on_daily_loss() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    rets = pd.Series([0.0] * 9 + [-0.05], index=idx)  # 마지막 날 -5%
    state = evaluate_kill_switch(rets, config=KillSwitchConfig(daily_loss_limit=-0.03))
    assert state.triggered
    assert "일일 손실" in state.reason


def test_kill_switch_triggers_on_cumulative_drawdown() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    # 점진적 하락 -25%
    rets = pd.Series([-0.014] * 20, index=idx)
    state = evaluate_kill_switch(rets, config=KillSwitchConfig(cumulative_loss_limit=-0.20))
    assert state.triggered
    assert "MDD" in state.reason


def test_kill_switch_triggers_on_consecutive_losses() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    # 5일 연속 -1%, 나머진 0
    vals = [0.0] * 10 + [-0.01] * 5 + [0.0] * 5
    rets = pd.Series(vals, index=idx)
    state = evaluate_kill_switch(
        rets,
        config=KillSwitchConfig(
            daily_loss_limit=-0.10,  # 안 발동
            cumulative_loss_limit=-0.50,  # 안 발동
            consecutive_losing_days=5,
            consecutive_loss_threshold=-0.005,
        ),
    )
    assert state.triggered
    assert "연속" in state.reason


# -----------------------------------------------------------------------------
# Kelly sizing
# -----------------------------------------------------------------------------
def test_quarter_kelly_zero_when_no_edge() -> None:
    assert quarter_kelly_fraction(0.0, 0.01) == 0.0


def test_quarter_kelly_zero_when_negative_edge() -> None:
    assert quarter_kelly_fraction(-0.01, 0.01) == 0.0


def test_quarter_kelly_capped() -> None:
    # full kelly = 5.0 → quarter = 1.25 → cap 0.5
    assert quarter_kelly_fraction(0.05, 0.01, cap=0.5) == 0.5


def test_quarter_kelly_normal() -> None:
    # full = 0.01/0.0001 = 100 → quarter = 25 → cap 0.5
    assert quarter_kelly_fraction(0.01, 0.0001, cap=0.5) == 0.5
    # full = 0.001/0.001 = 1 → quarter = 0.25
    assert abs(quarter_kelly_fraction(0.001, 0.001, cap=0.5) - 0.25) < 1e-9


# -----------------------------------------------------------------------------
# Exposure limits
# -----------------------------------------------------------------------------
def test_exposure_no_violations_under_limits() -> None:
    w = pd.Series({f"T{i}": 0.1 for i in range(10)})  # 10종목 각 10%
    violations = check_exposure(w)
    assert violations == []


def test_exposure_violates_single_position() -> None:
    w = pd.Series({"A": 0.5, "B": 0.5})  # A가 50% > 30%
    violations = check_exposure(w, ExposureLimits(max_single_position=0.30))
    assert any("단일 종목" in v for v in violations)


def test_exposure_violates_total() -> None:
    w = pd.Series({f"T{i}": 0.15 for i in range(10)})  # 합 1.5
    violations = check_exposure(w, ExposureLimits(max_total_exposure=1.0))
    assert any("총 노출" in v for v in violations)


def test_exposure_warns_concentration() -> None:
    w = pd.Series({"A": 0.25, "B": 0.20, "C": 0.20, "D": 0.05, "E": 0.05})
    # top3 = 65% > 50%
    violations = check_exposure(
        w, ExposureLimits(max_single_position=0.30, warn_concentration=0.50)
    )
    assert any("집중도" in v for v in violations)


# -----------------------------------------------------------------------------
# pre_trade_check 통합
# -----------------------------------------------------------------------------
def test_pre_trade_check_blocks_when_kill_switch_on() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    rets = pd.Series([0.0] * 9 + [-0.05], index=idx)
    weights = pd.Series({f"T{i}": 0.1 for i in range(10)})
    result = pre_trade_check(daily_returns=rets, target_weights=weights)
    assert not result.can_trade
    assert result.kill_switch.triggered


def test_pre_trade_check_allows_when_normal() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    rets = pd.Series(np.random.default_rng(1).normal(0.001, 0.005, 100), index=idx)
    weights = pd.Series({f"T{i}": 0.1 for i in range(10)})
    result = pre_trade_check(daily_returns=rets, target_weights=weights)
    assert result.can_trade
