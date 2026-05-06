"""리스크 관리 모듈.

라이브 트레이딩 전 필수 안전 장치 (가드레일 §9, §10).
백테스트에선 사후 검증용, 라이브에선 사전 차단용으로 동일 로직 사용.

구성:
1. Kill switch — 일일·누적 손실 한도 초과 시 신규 진입 차단
2. 켈리 사이징 — 1/4 켈리 (Full Kelly는 변동성 너무 큼)
3. 노출 한도 — 단일 종목 비중 상한, 섹터 집중 경고
4. MDD 한도 — 포트폴리오 MDD가 임계 도달 시 청산 신호
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.common.logger import logger


# ============================================================================
# Kill switch — 일일/누적 손실 한도
# ============================================================================
@dataclass(frozen=True)
class KillSwitchConfig:
    daily_loss_limit: float = -0.03  # 일중 -3% 도달 시 발동
    cumulative_loss_limit: float = -0.20  # 누적 -20% 시 발동
    consecutive_losing_days: int = 5  # 연속 손실 일수 임계
    consecutive_loss_threshold: float = -0.005  # "손실"로 간주할 일일 수익률


@dataclass
class KillSwitchState:
    triggered: bool
    reason: str
    triggered_at: pd.Timestamp | None


def evaluate_kill_switch(
    daily_returns: pd.Series,
    *,
    config: KillSwitchConfig | None = None,
) -> KillSwitchState:
    """Kill switch 발동 여부 평가 (백테스트 사후 검증).

    라이브에선 주문 직전 호출 — 현재 시점까지 returns 누적 후 판정.
    """
    cfg = config or KillSwitchConfig()
    rets = daily_returns.dropna()
    if len(rets) == 0:
        return KillSwitchState(triggered=False, reason="(데이터 없음)", triggered_at=None)

    # 1. 일일 손실 한도
    worst_daily = rets.min()
    if worst_daily <= cfg.daily_loss_limit:
        when = rets.index[rets.argmin()]
        return KillSwitchState(
            triggered=True,
            reason=f"일일 손실 한도 위반: {worst_daily * 100:.2f}% ≤ {cfg.daily_loss_limit * 100}%",
            triggered_at=when,
        )

    # 2. 누적 손실 (drawdown 기준)
    equity = (1.0 + rets).cumprod()
    dd = equity / equity.cummax() - 1.0
    worst_dd = dd.min()
    if worst_dd <= cfg.cumulative_loss_limit:
        when = dd.index[dd.argmin()]
        return KillSwitchState(
            triggered=True,
            reason=f"누적 MDD 한도 위반: {worst_dd * 100:.2f}% ≤ {cfg.cumulative_loss_limit * 100}%",
            triggered_at=when,
        )

    # 3. 연속 손실 일수
    losing = (rets <= cfg.consecutive_loss_threshold).astype(int)
    # rolling sum이 N에 도달한 적 있는지
    streak = losing.rolling(window=cfg.consecutive_losing_days).sum()
    if (streak >= cfg.consecutive_losing_days).any():
        when = streak.index[streak >= cfg.consecutive_losing_days][0]
        return KillSwitchState(
            triggered=True,
            reason=f"연속 {cfg.consecutive_losing_days}일 손실 (≤{cfg.consecutive_loss_threshold * 100}%/일)",
            triggered_at=when,
        )

    return KillSwitchState(triggered=False, reason="(정상)", triggered_at=None)


# ============================================================================
# 켈리 사이징 — 1/4 켈리
# ============================================================================
def quarter_kelly_fraction(
    expected_return: float,
    variance: float,
    *,
    cap: float = 0.5,
) -> float:
    """1/4 켈리 비율.

    Full Kelly = expected_return / variance.
    Full은 변동성이 너무 커서 실제로 30~50% drawdown 흔함.
    1/4 켈리는 학술 권장 — Sharpe 손실 약 1/4, 변동성 1/4로 대폭 감소.

    cap: 단일 자산 노출 상한 (디폴트 50%).
    """
    if variance <= 0:
        return 0.0
    full = expected_return / variance
    quarter = full / 4.0
    return float(max(0.0, min(quarter, cap)))


def position_sizes_from_returns(
    historical_returns: pd.DataFrame,
    *,
    lookback_days: int = 252,
    cap_per_asset: float = 0.30,
    total_cap: float = 1.0,
) -> pd.Series:
    """과거 N일 수익률 → 자산별 1/4 켈리 비중.

    historical_returns: 일별 자산 수익률 (columns=tickers).
    Returns: 자산별 비중 (합 ≤ total_cap).
    """
    recent = historical_returns.tail(lookback_days)
    means = recent.mean()
    vars_ = recent.var(ddof=0)

    sizes = pd.Series(
        {
            t: quarter_kelly_fraction(means[t], vars_[t], cap=cap_per_asset)
            for t in historical_returns.columns
        }
    )
    total = sizes.sum()
    if total > total_cap:
        sizes = sizes * (total_cap / total)
    return sizes


# ============================================================================
# 노출 한도 — 단일 종목·섹터
# ============================================================================
@dataclass(frozen=True)
class ExposureLimits:
    max_single_position: float = 0.30  # 단일 종목 30% 상한
    max_total_exposure: float = 1.00  # 전체 자본 100% (레버리지 금지)
    warn_concentration: float = 0.50  # 상위 N개 종목 합 > 50% 경고


def check_exposure(weights: pd.Series, limits: ExposureLimits | None = None) -> list[str]:
    """현재 포트폴리오 가중치가 한도를 위반하는지 검사."""
    lim = limits or ExposureLimits()
    violations: list[str] = []
    w = weights.dropna()

    # 단일 종목 상한
    over = w[w > lim.max_single_position]
    for t, v in over.items():
        violations.append(
            f"단일 종목 한도 초과: {t}={v * 100:.1f}% > {lim.max_single_position * 100}%"
        )

    # 전체 노출
    total = w.sum()
    if total > lim.max_total_exposure:
        violations.append(f"총 노출 초과: {total * 100:.1f}% > {lim.max_total_exposure * 100}%")

    # 집중도 경고
    top3 = w.nlargest(3).sum()
    if top3 > lim.warn_concentration:
        violations.append(
            f"⚠️ 상위 3종목 집중도 {top3 * 100:.1f}% > {lim.warn_concentration * 100}%"
        )

    return violations


# ============================================================================
# 통합 — 라이브 트레이딩 사전 점검
# ============================================================================
@dataclass(frozen=True)
class RiskCheckResult:
    can_trade: bool
    kill_switch: KillSwitchState
    exposure_violations: list[str]
    notes: list[str]


def pre_trade_check(
    *,
    daily_returns: pd.Series,
    target_weights: pd.Series,
    kill_config: KillSwitchConfig | None = None,
    exposure_limits: ExposureLimits | None = None,
) -> RiskCheckResult:
    """라이브 주문 직전 종합 점검.

    can_trade=False 면 어떤 새로운 매매도 금지. 청산만 가능.
    """
    ks = evaluate_kill_switch(daily_returns, config=kill_config)
    ev = check_exposure(target_weights, limits=exposure_limits)

    notes: list[str] = []
    can = not ks.triggered
    if not any("⚠️" in v for v in ev) and any("초과" in v for v in ev):
        # 한도 "초과"는 차단, "⚠️ 경고"는 허용
        can = False
        notes.append("노출 한도 위반 — 매매 차단")

    if ks.triggered:
        logger.error(f"🚨 Kill switch 발동: {ks.reason}")
        notes.append(f"Kill switch: {ks.reason}")
    for v in ev:
        if "⚠️" in v:
            logger.warning(v)
        else:
            logger.error(v)

    return RiskCheckResult(
        can_trade=can,
        kill_switch=ks,
        exposure_violations=ev,
        notes=notes,
    )
