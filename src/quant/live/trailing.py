"""트레일링 스톱 — 종목별 최고 손익률 추적.

매일 KIS profit_pct를 받아 종목별 최고치 누적. 현재가 최고치 대비
TRAIL_DRAWDOWN_PCT 이상 떨어지면 매도 트리거.

작동 조건:
  - 최고 손익률 ≥ TRAIL_ACTIVATION_PCT (예: +10% 이상 올랐던 종목)
  - 현재 손익률 ≤ (최고 - TRAIL_DRAWDOWN_PCT) (예: 최고에서 -10% 후퇴)

저장: data/state/trailing.json
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from quant.common.config import get_settings
from quant.common.logger import logger

# 종목 최고치에서 떨어진 % — 도달 시 매도
TRAIL_DRAWDOWN_PCT = 10.0
# 최고 손익률 활성화 임계 — 이 % 이상 올랐던 종목만 트레일링 적용
TRAIL_ACTIVATION_PCT = 10.0


def _trailing_path() -> Path:
    s = get_settings()
    p = s.data_dir / "state" / "trailing.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_trailing() -> dict[str, dict]:
    """저장된 트레일링 데이터 로드. 실패 시 빈 dict."""
    p = _trailing_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"trailing.json 로드 실패: {e}")
        return {}


def save_trailing(data: dict[str, dict]) -> None:
    p = _trailing_path()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_max(ticker: str, profit_pct: float, data: dict[str, dict]) -> None:
    """현재 profit_pct로 max_profit_pct 갱신 (in-place)."""
    today = date.today().isoformat()
    entry = data.get(ticker)
    if entry is None:
        data[ticker] = {"max_profit_pct": profit_pct, "last_update": today}
        return
    if profit_pct > entry["max_profit_pct"]:
        entry["max_profit_pct"] = profit_pct
    entry["last_update"] = today


def check_trail_trigger(
    ticker: str, profit_pct: float, data: dict[str, dict]
) -> tuple[bool, float | None]:
    """트레일 트리거 체크. (triggered, max_seen) 반환."""
    entry = data.get(ticker)
    if not entry:
        return (False, None)
    max_seen = entry["max_profit_pct"]
    if max_seen < TRAIL_ACTIVATION_PCT:
        return (False, max_seen)
    drawdown = max_seen - profit_pct
    if drawdown >= TRAIL_DRAWDOWN_PCT:
        return (True, max_seen)
    return (False, max_seen)


def prune_stale(data: dict[str, dict], held_tickers: set[str]) -> None:
    """보유 안 한 종목 데이터 제거 (in-place)."""
    for ticker in list(data.keys()):
        if ticker not in held_tickers:
            del data[ticker]
