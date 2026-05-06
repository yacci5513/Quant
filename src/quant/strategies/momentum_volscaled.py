"""변동성 조정 모멘텀 (Volatility-Scaled Momentum).

기본 모멘텀 Top-N과 같은 종목 선정이지만, 가중치를 변동성 역수로 조정.
변동성이 큰 종목에 적게, 안정적인 종목에 많이 분배 → Sharpe 개선·MDD 감소 기대.

학술 근거: Asness, Moskowitz, Pedersen — Value and Momentum Everywhere (2013)
가드레일 §1: 추가 파라미터 1개(vol_window)만 → 과적합 위험 미미.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.common.logger import logger


@dataclass(frozen=True)
class MomentumVolScaledConfig:
    lookback_months: int = 12
    skip_months: int = 1
    top_n: int = 10
    vol_window_days: int = 60  # 변동성 측정 윈도우
    min_avg_value: float = 1e9
    rebalance_freq: str = "BMS"


def _momentum_score(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    end = prices.shift(skip * 21)
    start = prices.shift(lookback * 21)
    return end / start - 1.0


def _rolling_vol(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """일별 수익률의 N일 표준편차 (연환산하지 않음 — 상대 비교만)."""
    return prices.pct_change().rolling(window=window, min_periods=20).std()


def _liquidity_mask(value_panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    avg = value_panel.rolling(window=60, min_periods=20).mean()
    return avg >= threshold


def generate_weights(
    prices: pd.DataFrame,
    *,
    values: pd.DataFrame | None = None,
    config: MomentumVolScaledConfig | None = None,
) -> pd.DataFrame:
    cfg = config or MomentumVolScaledConfig()
    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns)

    momentum = _momentum_score(prices, cfg.lookback_months, cfg.skip_months)
    vol = _rolling_vol(prices, cfg.vol_window_days)

    valid = prices.notna() & (prices > 0) & vol.notna() & (vol > 0)
    if values is not None:
        valid = valid & _liquidity_mask(values, cfg.min_avg_value)
    momentum = momentum.where(valid)

    rebalance_dates = pd.date_range(
        start=prices.index[0], end=prices.index[-1], freq=cfg.rebalance_freq
    )
    rb_idx = []
    for d in rebalance_dates:
        future = prices.index[prices.index >= d]
        if len(future) > 0:
            rb_idx.append(future[0])
    rb_idx = pd.DatetimeIndex(sorted(set(rb_idx)))

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)
    skipped = 0
    for d in rb_idx:
        scores = momentum.loc[d].dropna()
        if len(scores) < cfg.top_n:
            skipped += 1
            continue
        top = scores.nlargest(cfg.top_n).index
        # 변동성 역수 가중치 (inverse vol weighting)
        vols = vol.loc[d, top].fillna(vol.loc[d, top].mean())
        inv_vol = 1.0 / vols
        # 정규화
        w_norm = inv_vol / inv_vol.sum()
        weights.loc[d, :] = 0.0
        for t, wv in w_norm.items():
            weights.loc[d, t] = float(wv)

    weights_at_rb = weights.loc[rb_idx]
    weights_full = weights_at_rb.reindex(prices.index, method="ffill").fillna(0.0)

    if skipped > 0:
        logger.warning(f"momentum_volscaled: {skipped}/{len(rb_idx)} 리밸런싱 스킵")
    logger.info(f"momentum_volscaled: rebalances={len(rb_idx) - skipped}, top_n={cfg.top_n}")
    return weights_full


def _ignore() -> None:
    _ = np
