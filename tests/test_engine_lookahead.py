"""백테스트 엔진 lookahead 자동 보호 테스트.

가드레일 §3: weights[t]는 t 종가로 계산되므로 t+1 수익에만 반영되어야 함.
auto_shift_weights=True (기본)이 이 보호를 자동 적용.
"""

from __future__ import annotations

import pandas as pd

from quant.backtest.costs import BLUECHIP_KIS
from quant.backtest.engine import run_backtest


def _two_day_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """2영업일짜리 합성 패널 — lookahead 효과를 명시적으로 측정."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100.0, 110.0, 120.0]}, index=idx)
    # weights[t=0]=1.0 → 전 기간 보유. shift 후엔 t=1부터 적용
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    return prices, weights


def test_default_shift_protects_first_day() -> None:
    """auto_shift_weights=True (기본) — 첫 날엔 보유 0 (shift로 NaN→0)."""
    prices, weights = _two_day_panel()
    res = run_backtest(prices, weights, cost_model=BLUECHIP_KIS)
    # 첫 날 수익률 0 (보유 0이라)
    assert abs(res.daily_returns.iloc[0]) < 1e-9


def test_opt_out_uses_same_day_lookahead() -> None:
    """auto_shift_weights=False — 가중치 변하는 시나리오에서 lookahead 효과 측정.

    시그널이 t에 신고가 갱신을 보고 t 가중치 1.0 → t 수익률 그대로 캡쳐 (lookahead).
    shift 자동 보호 시엔 t+1에 적용 → 이미 가격이 떨어져서 손실 가능.
    """
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"A": [100.0, 100.0, 120.0, 110.0]}, index=idx)
    # 3일째 신고가 → 그날 가중치 1.0, 다음 날 청산
    weights = pd.DataFrame({"A": [0.0, 0.0, 1.0, 0.0]}, index=idx)

    res_unsafe = run_backtest(prices, weights, cost_model=BLUECHIP_KIS, auto_shift_weights=False)
    res_safe = run_backtest(prices, weights, cost_model=BLUECHIP_KIS, auto_shift_weights=True)
    cum_unsafe = res_unsafe.daily_returns.add(1).prod() - 1
    cum_safe = res_safe.daily_returns.add(1).prod() - 1
    # unsafe: 3일째 +20% 캡쳐 (lookahead)
    # safe: 4일째에 적용, 가격 -8.3% → 손실
    assert cum_unsafe > 0.15
    assert cum_safe < 0


def test_lookahead_inflates_breakout_strategy() -> None:
    """breakout-like 시그널 — shift 자동 보호 시 누출 차단 검증.

    가짜 시나리오: 종가가 신고가 갱신 → 그날 가중치 1.0.
    shift=False 면 그날 종가 정보로 그날 매수 → 그날 수익 100% 캡쳐 (lookahead).
    shift=True 면 다음 날 매수 → 신고가 갱신 후 가격이 떨어지면 손실 (정상).
    """
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    # 5일째 가격 급등 후 다음날 떨어짐 → lookahead가 잡으면 +20% 그대로 받음
    prices = pd.DataFrame(
        {"A": [100, 100, 100, 100, 120, 110, 105, 105, 105, 105]}, index=idx, dtype=float
    )
    # 신고가 갱신일(5일째)에만 가중치 1.0 → 그 다음날 가중치 0
    weights = pd.DataFrame({"A": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]}, index=idx, dtype=float)

    res_with_lookahead = run_backtest(
        prices, weights, cost_model=BLUECHIP_KIS, auto_shift_weights=False
    )
    res_safe = run_backtest(prices, weights, cost_model=BLUECHIP_KIS, auto_shift_weights=True)

    # lookahead: 5일째에 +20% 캡쳐 (전일 100 → 120 = +20%)
    cum_lookahead = res_with_lookahead.daily_returns.add(1).prod() - 1
    cum_safe = res_safe.daily_returns.add(1).prod() - 1
    # safe는 6일째에 매수 → 110에서 105로 떨어짐 = -4.5%
    assert cum_lookahead > 0.15  # 위양성 ↑
    assert cum_safe < 0  # 정상 (다음날 매수, 떨어진 가격 받음)
