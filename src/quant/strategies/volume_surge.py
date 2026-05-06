"""거래량 급증 (Volume Surge) 전략.

근거: William O'Neil (CAN SLIM), Wyckoff Volume Spread Analysis
- 거래량 5일 평균이 60일 평균보다 크게 높으면 매집·관심 급증 신호
- 가격 상승과 결합 시 모멘텀 초입 시점 포착 가능

규칙:
1. ratio = vol_5d_mean / vol_60d_mean
2. ratio ≥ 임계 (기본 2.0배) + 가격 상승 (5일 수익률 > 0) 종목 Top-N
3. 동일가중 또는 ratio 기반 가중

가드레일:
§1 과적합:   파라미터 2개 (short_window, long_window)
§3 미래참조: rolling은 t 이전 데이터만
§4 거래비용: 회전율 큼 — 단독 사용보다 결합 권장
§6 데이터품질: 0거래량 종목 자동 제외
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.common.logger import logger


@dataclass(frozen=True)
class VolumeSurgeConfig:
    short_window: int = 5  # 단기 평균
    long_window: int = 60  # 장기 평균
    ratio_threshold: float = 2.0  # 단/장 비율 임계 (2.0 = 단기가 장기의 2배)
    price_rise_window: int = 5  # 가격 상승 확인 윈도우 (5일)
    top_n: int = 10
    min_avg_value: float = 1e9
    rebalance_freq: str = "BMS"


def _volume_surge_score(
    prices: pd.DataFrame, volumes: pd.DataFrame, cfg: VolumeSurgeConfig
) -> pd.DataFrame:
    """거래량 단기/장기 비율 × 가격 상승 시그널."""
    short_vol = volumes.rolling(window=cfg.short_window, min_periods=cfg.short_window).mean()
    long_vol = volumes.rolling(window=cfg.long_window, min_periods=20).mean()
    ratio = short_vol / long_vol.where(long_vol > 0)

    price_change = prices.pct_change(periods=cfg.price_rise_window)
    # 점수 = 거래량 비율 × 가격 상승률 (둘 다 양수일 때 의미)
    score = ratio * (1 + price_change.clip(lower=0))
    # 임계 미달은 NaN
    score = score.where(ratio >= cfg.ratio_threshold)
    score = score.where(price_change > 0)
    return score


def _liquidity_mask(value_panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    avg = value_panel.rolling(window=60, min_periods=20).mean()
    return avg >= threshold


def generate_weights(
    prices: pd.DataFrame,
    *,
    volumes: pd.DataFrame | None = None,
    values: pd.DataFrame | None = None,
    config: VolumeSurgeConfig | None = None,
) -> pd.DataFrame:
    """거래량 급증 + 가격 상승 종목 Top-N."""
    cfg = config or VolumeSurgeConfig()
    if prices.empty or volumes is None:
        return pd.DataFrame()

    score = _volume_surge_score(prices, volumes, cfg)
    valid = prices.notna() & (prices > 0)
    if values is not None:
        valid = valid & _liquidity_mask(values, cfg.min_avg_value)
    candidates = score.where(valid)

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
        scores = candidates.loc[d].dropna()
        if len(scores) < cfg.top_n:
            skipped += 1
            continue
        top = scores.nlargest(cfg.top_n).index
        weights.loc[d, :] = 0.0
        w = 1.0 / cfg.top_n
        for t in top:
            weights.loc[d, t] = w

    weights_at_rb = weights.loc[rb_idx]
    weights_full = weights_at_rb.reindex(prices.index, method="ffill").fillna(0.0)
    if skipped > 0:
        logger.warning(
            f"volume_surge: {skipped}/{len(rb_idx)} 리밸런싱 스킵 (조건 충족 종목 < {cfg.top_n})"
        )
    logger.info(f"volume_surge: rebalances={len(rb_idx) - skipped}, top_n={cfg.top_n}")
    return weights_full
