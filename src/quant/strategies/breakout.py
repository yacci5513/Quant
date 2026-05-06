"""52주 신고가 돌파 (Donchian breakout) 전략.

근거: Donchian (1960), Turtle Traders (1980s), William O'Neil CAN SLIM
- 종가가 과거 N일 신고가를 돌파하면 추세 시작 신호
- 모멘텀(이미 오른 종목)과 다르게 추세 초기를 포착할 가능성

규칙:
1. 매 리밸런싱 일자에 close[t] / max(close[t-window:t]) 비율 계산
2. 비율 = 1.0 (신고가) 또는 그에 근접한 종목 Top-N 선정
3. 동일가중

가드레일:
§1 과적합:   파라미터 1개(window) — 단순함
§3 미래참조: rolling().max()는 t 이전 데이터만
§5 Crowding: 단순 시그널이라 widely known. 모멘텀과 결합 시 분산 효과
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.common.logger import logger


@dataclass(frozen=True)
class BreakoutConfig:
    window_days: int = 252  # 약 1년 영업일 = 52주
    top_n: int = 10  # 선정 종목
    min_avg_value: float = 1e9  # 일평균 거래대금 임계
    rebalance_freq: str = "BMS"  # 월초 영업일 (기본)
    proximity: float = 0.98  # 신고가 비율 임계 (0.98 = 신고가의 98% 이상)


def _high_proximity(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """현재 종가 / 과거 N일 최고가."""
    rolling_max = prices.rolling(window=window, min_periods=max(20, window // 4)).max()
    return prices / rolling_max


def _liquidity_mask(value_panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    avg = value_panel.rolling(window=60, min_periods=20).mean()
    return avg >= threshold


def generate_weights(
    prices: pd.DataFrame,
    *,
    values: pd.DataFrame | None = None,
    config: BreakoutConfig | None = None,
) -> pd.DataFrame:
    """52주 신고가 근접 Top-N 동일가중."""
    cfg = config or BreakoutConfig()
    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns)

    proximity = _high_proximity(prices, cfg.window_days)
    valid = prices.notna() & (prices > 0) & (proximity >= cfg.proximity)
    if values is not None:
        valid = valid & _liquidity_mask(values, cfg.min_avg_value)

    candidates = proximity.where(valid)

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
        # 신고가에 가장 가까운 N개 = 가장 큰 점수
        top = scores.nlargest(cfg.top_n).index
        weights.loc[d, :] = 0.0
        w = 1.0 / cfg.top_n
        for t in top:
            weights.loc[d, t] = w

    weights_at_rb = weights.loc[rb_idx]
    weights_full = weights_at_rb.reindex(prices.index, method="ffill").fillna(0.0)
    if skipped > 0:
        logger.warning(
            f"breakout: {skipped}/{len(rb_idx)} 리밸런싱 스킵 (신고가 종목 < {cfg.top_n})"
        )
    logger.info(
        f"breakout: rebalances={len(rb_idx) - skipped}, top_n={cfg.top_n}, window={cfg.window_days}일"
    )
    return weights_full
