"""Kill switch 2일 연속 트리거 검증."""

from __future__ import annotations

from datetime import date

from quant.live.killswitch import KILL_SWITCH_PCT, evaluate


def test_normal_no_armed():
    """손익률이 임계 위면 armed 안 됨."""
    d, state = evaluate(date(2026, 5, 17), -10.0, state={})
    assert not d.armed
    assert not d.should_fire
    assert state == {}


def test_first_violation_armed_only():
    """첫 위반 — armed 상태만 (발동 X)."""
    d, state = evaluate(date(2026, 5, 17), -22.0, state={})
    assert d.armed
    assert not d.should_fire
    assert state["armed_date"] == "2026-05-17"


def test_second_consecutive_day_fires():
    """어제 armed + 오늘도 위반 → 발동."""
    prev = {"armed_date": "2026-05-16", "armed_loss_pct": -21.0}
    d, state = evaluate(date(2026, 5, 17), -23.0, state=prev)
    assert d.should_fire
    assert "fired_date" in state


def test_recovery_disarms():
    """armed 후 오늘 회복 → armed 해제, 발동 안 함."""
    prev = {"armed_date": "2026-05-16", "armed_loss_pct": -21.0}
    d, state = evaluate(date(2026, 5, 17), -10.0, state=prev)
    assert not d.armed
    assert not d.should_fire
    assert state == {}


def test_weekend_gap_within_3_days():
    """금요일 armed → 월요일 위반 (3일 갭) → 발동."""
    prev = {"armed_date": "2026-05-15", "armed_loss_pct": -21.0}  # 금
    d, state = evaluate(date(2026, 5, 18), -22.0, state=prev)  # 월
    assert d.should_fire


def test_too_old_armed_resets():
    """4일 이상 갭이면 새로 armed (이전 armed 무효)."""
    prev = {"armed_date": "2026-05-10", "armed_loss_pct": -21.0}
    d, state = evaluate(date(2026, 5, 17), -22.0, state=prev)
    assert d.armed
    assert not d.should_fire  # 새로 armed
    assert state["armed_date"] == "2026-05-17"


def test_none_loss_pct_no_decision():
    """잔고 미연동 — 보류."""
    d, state = evaluate(date(2026, 5, 17), None, state={})
    assert not d.armed
    assert not d.should_fire


def test_threshold_default():
    """기본 임계 -20%."""
    assert KILL_SWITCH_PCT == -20.0
    d, _ = evaluate(date(2026, 5, 17), -19.9, state={})
    assert not d.armed  # -19.9% 는 > -20% 이므로 정상
    d, _ = evaluate(date(2026, 5, 17), -20.1, state={})
    assert d.armed
