"""다중 레짐 전략 결합 — 시장 상황별 다른 시그널 적용.

레짐 분류 (단순화):
  BULL_STRONG  : 시장 > MA100 + 단기 추세 강함 (5d > 0)
  BULL_CHOPPY  : 시장 > MA100 + 단기 변동성 높음 (5d 표준편차 > 임계)
  BEAR         : 시장 < MA100
  RECOVERY     : 어제까지 BEAR → 오늘 BULL 진입 직후 (3일 윈도우)

레짐별 가중치 정책 (실험적):
  BULL_STRONG  → 100% 모멘텀 (현재 챔피언)
  BULL_CHOPPY  → 60% 모멘텀 + 40% 저변동성 (변동성 흡수)
  BEAR         →   0% (현금 100%)
  RECOVERY     →  50% 모멘텀 + 50% 현금 (점진적 진입)

단순함의 가치: 4 레짐 × 1~2개 시그널 = 일일 분기 룰.
복잡한 ML 분류기보다 견고하다 (Karpathy 원칙).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from quant.backtest.regime import market_proxy
from quant.common.logger import logger


class Regime(str, Enum):
    BULL_STRONG = "bull_strong"  # 강세 + 추세 강함
    BULL_CHOPPY = "bull_choppy"  # 강세 + 변동성 높음
    BEAR = "bear"  # 약세
    RECOVERY = "recovery"  # 회복 직후


@dataclass(frozen=True)
class RegimeConfig:
    ma_window: int = 100  # 시장 MA 윈도우
    short_window: int = 5  # 단기 추세 측정
    chop_vol_threshold: float = 0.015  # 일일 표준편차 임계 (1.5%)
    recovery_days: int = 3  # MA 회복 후 RECOVERY로 분류할 기간


def classify_regimes(prices: pd.DataFrame, cfg: RegimeConfig | None = None) -> pd.Series:
    """일별 시장 레짐 분류.

    Returns:
        Series with index=date, values=Regime.
    """
    cfg = cfg or RegimeConfig()
    proxy = market_proxy(prices)
    ma = proxy.rolling(cfg.ma_window, min_periods=max(20, cfg.ma_window // 2)).mean()

    above_ma = proxy > ma
    short_ret = proxy.pct_change(periods=cfg.short_window)
    short_vol = proxy.pct_change().rolling(cfg.short_window).std()

    # 분류
    regime = pd.Series(Regime.BEAR.value, index=proxy.index, dtype="object")
    regime[above_ma & (short_ret > 0) & (short_vol < cfg.chop_vol_threshold)] = (
        Regime.BULL_STRONG.value
    )
    regime[above_ma & ((short_ret <= 0) | (short_vol >= cfg.chop_vol_threshold))] = (
        Regime.BULL_CHOPPY.value
    )
    regime[~above_ma] = Regime.BEAR.value

    # RECOVERY: 어제까지 BEAR → 오늘 BULL_*
    yesterday_bear = (~above_ma).shift(1).fillna(False)
    today_bull = above_ma
    just_recovered = yesterday_bear & today_bull
    # 회복 후 N영업일 동안 RECOVERY로 표시
    recovery_mask = just_recovered.rolling(cfg.recovery_days, min_periods=1).max().astype(bool)
    regime[recovery_mask & today_bull] = Regime.RECOVERY.value

    logger.info(
        f"레짐 분류 완료: "
        f"BULL_STRONG {(regime == Regime.BULL_STRONG.value).sum()}일, "
        f"BULL_CHOPPY {(regime == Regime.BULL_CHOPPY.value).sum()}일, "
        f"BEAR {(regime == Regime.BEAR.value).sum()}일, "
        f"RECOVERY {(regime == Regime.RECOVERY.value).sum()}일"
    )
    return regime


@dataclass(frozen=True)
class RegimePolicy:
    """레짐별 가중치 분배 정책.

    각 dict의 키는 시그널 패널 이름, 값은 그 시그널의 비중 (합 1.0 또는 0).
    합이 0이면 해당 레짐엔 현금 보유.
    """

    bull_strong: dict[str, float]
    bull_choppy: dict[str, float]
    bear: dict[str, float]
    recovery: dict[str, float]


# 권장 정책 (실험 시작점 — 백테스트로 튜닝 가능)
DEFAULT_POLICY = RegimePolicy(
    bull_strong={"momentum": 1.0},
    bull_choppy={"momentum": 0.6, "low_vol": 0.4},
    bear={},  # 현금 100%
    recovery={"momentum": 0.5},  # 50% 진입, 50% 현금
)


def combine_by_regime(
    prices: pd.DataFrame,
    signal_panels: dict[str, pd.DataFrame],
    *,
    policy: RegimePolicy | None = None,
    regime_config: RegimeConfig | None = None,
) -> pd.DataFrame:
    """일별 레짐 → 정책 → 가중치 패널 결합.

    Args:
        prices: 가격 패널 (레짐 분류용 시장 proxy)
        signal_panels: {"momentum": panel, "low_vol": panel, ...}
        policy: 레짐별 가중치 정책 (None이면 DEFAULT_POLICY)
        regime_config: 레짐 분류 파라미터

    Returns:
        결합 가중치 패널 (각 일자 사용 정책에 따라 분기 적용).
    """
    pol = policy or DEFAULT_POLICY
    regimes = classify_regimes(prices, regime_config)

    regime_to_weights = {
        Regime.BULL_STRONG.value: pol.bull_strong,
        Regime.BULL_CHOPPY.value: pol.bull_choppy,
        Regime.BEAR.value: pol.bear,
        Regime.RECOVERY.value: pol.recovery,
    }

    # 합집합 컬럼
    all_cols = sorted(set().union(*[p.columns for p in signal_panels.values()]))
    combined = pd.DataFrame(0.0, index=prices.index, columns=all_cols, dtype=float)

    # 일자별 정책 적용
    for d in prices.index:
        regime_today = regimes.loc[d] if d in regimes.index else Regime.BEAR.value
        allocs = regime_to_weights.get(regime_today, {})
        if not allocs:
            continue  # 현금 보유 — combined 행 0 그대로
        total = sum(allocs.values())
        if total <= 0:
            continue
        for sig_name, weight in allocs.items():
            panel = signal_panels.get(sig_name)
            if panel is None or d not in panel.index:
                continue
            row = panel.loc[d].reindex(all_cols).fillna(0.0)
            # 시그널 패널 자체 합 정규화 (Top-N 동일가중이라 sum=1)
            row_sum = row.sum()
            if row_sum > 0:
                row = row / row_sum
            combined.loc[d] = combined.loc[d] + (weight / total) * row

    return combined


def regime_distribution(regimes: pd.Series) -> dict[str, float]:
    """레짐별 일수 비율."""
    counts = regimes.value_counts()
    total = counts.sum()
    return {k: float(v / total) for k, v in counts.items()}


def _ignore() -> None:
    _ = np
