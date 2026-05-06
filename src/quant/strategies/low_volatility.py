"""저변동성 (Low-Volatility) 전략.

규칙:
1. 매 리밸런싱 일자에 60일 일별 수익률 표준편차 하위 N종목 선정
2. 단, 12개월 모멘텀이 양(+)인 종목만 (장기 약세 종목 제외)
3. 동일가중 또는 inverse-vol 가중

학술 근거:
- Frazzini & Pedersen (2014) "Betting Against Beta" — 저변동성 알파의 존재 증명
- Asness, Frazzini, Pedersen "Low-Risk Investing Without Industry Bets"

가드레일:
§1 과적합     : 파라미터 2개(vol_window, top_n)
§2 생존편향   : 시점별 NaN 마스크
§3 미래참조   : 변동성은 t 이전 데이터만 사용
§4 거래비용   : 회전율 자체는 모멘텀과 비슷 (2~3x/년)
§5 Crowding   : 알려진 알파 — Crowding 진행 중. 모멘텀과 결합 시 분산 효과
§6 데이터품질 : 0거래량/NaN 자동 제외
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.common.logger import logger


@dataclass(frozen=True)
class LowVolatilityConfig:
    vol_window_days: int = 60  # 변동성 측정 윈도우
    long_lookback_months: int = 12  # 장기 모멘텀 필터 (음수면 제외)
    top_n: int = 10  # 선정 종목 수
    min_avg_value: float = 1e9  # 일평균 거래대금 필터
    rebalance_freq: str = "BMS"  # 월초 영업일
    inverse_vol_weight: bool = False  # True = 변동성 역수 가중, False = 동일가중


def _rolling_vol(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return prices.pct_change().rolling(window=window, min_periods=20).std()


def _long_momentum(prices: pd.DataFrame, months: int) -> pd.DataFrame:
    return prices / prices.shift(months * 21) - 1.0


def _liquidity_mask(value_panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    avg = value_panel.rolling(window=60, min_periods=20).mean()
    return avg >= threshold


def generate_weights(
    prices: pd.DataFrame,
    *,
    values: pd.DataFrame | None = None,
    config: LowVolatilityConfig | None = None,
) -> pd.DataFrame:
    """저변동성 종목 동일/역변동 가중 패널."""
    cfg = config or LowVolatilityConfig()
    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns)

    vol = _rolling_vol(prices, cfg.vol_window_days)
    long_mom = _long_momentum(prices, cfg.long_lookback_months)

    valid = prices.notna() & (prices > 0) & vol.notna() & (vol > 0) & (long_mom > 0)
    if values is not None:
        valid = valid & _liquidity_mask(values, cfg.min_avg_value)

    candidates = vol.where(valid)

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
        # 변동성 작은 N개 선정
        bottom = scores.nsmallest(cfg.top_n).index

        weights.loc[d, :] = 0.0
        if cfg.inverse_vol_weight:
            v = vol.loc[d, bottom]
            inv = 1.0 / v
            inv = inv / inv.sum()
            for t, wv in inv.items():
                weights.loc[d, t] = float(wv)
        else:
            w = 1.0 / cfg.top_n
            for t in bottom:
                weights.loc[d, t] = w

    weights_at_rb = weights.loc[rb_idx]
    weights_full = weights_at_rb.reindex(prices.index, method="ffill").fillna(0.0)

    if skipped > 0:
        logger.warning(f"low_volatility: {skipped}/{len(rb_idx)} 리밸런싱 스킵")
    logger.info(f"low_volatility: rebalances={len(rb_idx) - skipped}, top_n={cfg.top_n}")
    return weights_full
