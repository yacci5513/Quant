"""포트폴리오 Kill switch — 2일 연속 임계 위반 시 발동.

설계:
- 매일 잔고 누적 손익률을 확인 (KIS API 응답 기준).
- 임계 (-20%) 이하면 첫째 날 "armed", 다음 영업일에도 이하면 발동.
- 회복하면 즉시 disarm — 단발 노이즈에 발동 안 함.

발동 시 동작 (execute.py):
- 손실 종목(profit_pct < 0)만 매도. 이익 종목은 보존.
- 단순 _sell_all과 다름 — +20% 종목까지 던지지 않음.

저장: data/state/killswitch.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from quant.common.config import get_settings
from quant.common.logger import logger

# 포트폴리오 누적 손익률 임계 (%) — 2일 연속 이하면 발동
KILL_SWITCH_PCT = -20.0
# 어제 ~ 며칠 전까지를 "연속"으로 인정 (주말 포함 한도)
MAX_GAP_DAYS = 3


@dataclass
class KillSwitchDecision:
    should_fire: bool
    armed: bool
    armed_date: date | None
    today_loss_pct: float | None
    reason: str


def _state_path() -> Path:
    s = get_settings()
    p = s.data_dir / "state" / "killswitch.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_state() -> dict:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"killswitch.json 로드 실패: {e}")
        return {}


def save_state(state: dict) -> None:
    p = _state_path()
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(
    today: date,
    current_loss_pct: float | None,
    *,
    threshold: float = KILL_SWITCH_PCT,
    state: dict | None = None,
) -> tuple[KillSwitchDecision, dict]:
    """오늘 손익률과 저장된 armed 상태로 발동 여부 판정.

    Returns:
        (decision, new_state) — new_state는 save_state로 저장.
    """
    if state is None:
        state = load_state()

    if current_loss_pct is None:
        return (
            KillSwitchDecision(False, False, None, None, "잔고 미연동 — 판정 보류"),
            state,
        )

    # 회복 — armed 클리어
    if current_loss_pct > threshold:
        if state.get("armed_date"):
            logger.info(f"Kill switch disarm (회복: {current_loss_pct:+.2f}% > {threshold}%)")
        return (
            KillSwitchDecision(
                False, False, None, current_loss_pct, f"정상 ({current_loss_pct:+.2f}%)"
            ),
            {},
        )

    # 임계 위반 상태
    armed_date_str = state.get("armed_date")
    if not armed_date_str:
        new_state = {"armed_date": today.isoformat(), "armed_loss_pct": current_loss_pct}
        return (
            KillSwitchDecision(
                False,
                True,
                today,
                current_loss_pct,
                f"⚠️ Armed (오늘 {current_loss_pct:+.2f}% ≤ {threshold}%) — 내일도 같으면 발동",
            ),
            new_state,
        )

    armed_date = date.fromisoformat(armed_date_str)
    gap = (today - armed_date).days
    if gap == 0:
        # 같은 날 재호출 — 상태 그대로
        return (
            KillSwitchDecision(False, True, armed_date, current_loss_pct, "Armed (같은 날 재호출)"),
            state,
        )
    if 1 <= gap <= MAX_GAP_DAYS:
        # 2일째 연속 위반 → 발동
        new_state = {"fired_date": today.isoformat(), "fired_loss_pct": current_loss_pct}
        return (
            KillSwitchDecision(
                True,
                True,
                armed_date,
                current_loss_pct,
                f"🚨 KILL ({armed_date} → {today} 2일 연속 ≤ {threshold}%, 오늘 {current_loss_pct:+.2f}%)",
            ),
            new_state,
        )
    # gap > MAX_GAP_DAYS — 너무 오래 전 armed, 새로 시작
    new_state = {"armed_date": today.isoformat(), "armed_loss_pct": current_loss_pct}
    return (
        KillSwitchDecision(
            False,
            True,
            today,
            current_loss_pct,
            f"⚠️ Re-armed (이전 armed {gap}일 전 — 갭 초과, 오늘로 재시작)",
        ),
        new_state,
    )
