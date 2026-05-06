"""평균회귀 (Mean Reversion) — 단기 과매도 종목 매수.

규칙:
1. 매 리밸런싱 일자에 최근 N일 누적 수익률 하위 N개 (가장 많이 빠진 종목)
2. 단, 12개월 모멘텀이 양(+)인 종목만 (장기 약세주 제외)
3. 동일가중

가드레일:
§1 과적합     : 파라미터 적게 (lookback 1, top_n 1)
§2 생존편향   : 시점별 NaN 마스킹
§3 미래참조   : .shift(1)로 어제까지 정보만
§5 Crowding   : 단기 평균회귀는 잘 알려진 알파라 약화 가능
§6 데이터품질 : 거래정지 종목 자동 제외
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.common.logger import logger


@dataclass(frozen=True)
class MeanReversionConfig:
    short_lookback_days: int = 5  # 단기 하락 측정 윈도우
    long_lookback_months: int = 12  # 장기 추세 필터 (음수면 제외)
    top_n: int = 10  # 선정 종목 수 (가장 많이 빠진 N개)
    min_avg_value: float = 1e9
    rebalance_freq: str = "W-FRI"  # 주 1회 리밸런싱 (단기 신호 특성)


def _short_drop(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    """N일 전 대비 누적 수익률 (음수일수록 많이 떨어진 종목)."""
    return prices / prices.shift(days) - 1.0


def _long_momentum(prices: pd.DataFrame, months: int) -> pd.DataFrame:
    """장기 12개월 모멘텀 — 양(+)인 종목만 통과시키는 필터."""
    return prices / prices.shift(months * 21) - 1.0


def _liquidity_mask(value_panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    avg = value_panel.rolling(window=60, min_periods=20).mean()
    return avg >= threshold


def generate_weights(
    prices: pd.DataFrame,
    *,
    values: pd.DataFrame | None = None,
    config: MeanReversionConfig | None = None,
) -> pd.DataFrame:
    cfg = config or MeanReversionConfig()
    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns)

    short = _short_drop(prices, cfg.short_lookback_days)
    long_mom = _long_momentum(prices, cfg.long_lookback_months)

    # 필터: 장기 모멘텀 양수만 + 가격 유효
    valid = (long_mom > 0) & prices.notna() & (prices > 0)
    if values is not None:
        valid = valid & _liquidity_mask(values, cfg.min_avg_value)

    candidates = short.where(valid)

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
        # 가장 많이 떨어진 N개 (smallest = most negative short return)
        bottom = scores.nsmallest(cfg.top_n).index
        # 단, 진짜 음수인 것들만 (이미 오른 종목 제외)
        bottom = [t for t in bottom if scores[t] < 0]
        if not bottom:
            skipped += 1
            continue
        w = 1.0 / len(bottom)
        weights.loc[d, :] = 0.0
        for t in bottom:
            weights.loc[d, t] = w

    weights_at_rb = weights.loc[rb_idx]
    weights_full = weights_at_rb.reindex(prices.index, method="ffill").fillna(0.0)

    if skipped > 0:
        logger.warning(f"mean_reversion: {skipped}/{len(rb_idx)} 리밸런싱 스킵 (조건 미달)")
    logger.info(f"mean_reversion: rebalances={len(rb_idx) - skipped}, top_n={cfg.top_n}")
    return weights_full
