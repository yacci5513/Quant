"""시장 레짐 필터 (MA200 기반 약세장 방어).

근거: Faber (2007) "A Quantitative Approach to Tactical Asset Allocation"
 — 200일 이평선 단순 룰만으로 MDD 절반 가까이 감소, Sharpe 크게 개선.

규칙:
- 시장 proxy = 보유 종목의 동일가중 평균 (KOSPI EW 200 근사)
- proxy > MA200 → 풀투자 (exposure = 1.0)
- proxy < MA200 → 청산 (binary) 또는 50% 디레버리지 (half)

가드레일:
§3 미래참조: rolling MA는 t 이전 데이터만 사용. .shift는 호출 측 책임.
§7 통계유의성: 5년에 약세장 사례 1~2번 → walk-forward로 견고성 추가 검증 필요.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd


class RegimeMode(str, Enum):
    BINARY = "binary"  # MA200 위 = 1.0, 아래 = 0.0
    HALF = "half"  # MA200 위 = 1.0, 아래 = 0.5
    OFF = "off"  # 필터 비활성 (항상 1.0)


def market_proxy(prices: pd.DataFrame) -> pd.Series:
    """보유 가능 종목 종가의 동일가중 평균을 시장 proxy로."""
    if prices.empty:
        return pd.Series(dtype=float)
    return prices.mean(axis=1)


def regime_exposure(
    market: pd.Series,
    *,
    ma_window: int = 200,
    mode: str | RegimeMode = RegimeMode.BINARY,
    min_periods_ratio: float = 0.5,
) -> pd.Series:
    """시장 proxy → 일별 노출 비율 (0.0 ~ 1.0).

    Args:
        market: 시장 proxy 시계열
        ma_window: 이동평균 윈도우(영업일). 기본 200일
        mode: BINARY (0/1) | HALF (0.5/1) | OFF (항상 1)
        min_periods_ratio: 초기 ma_window * ratio 일은 NaN 보호

    Returns:
        exposure 시리즈 (가중치에 곱하면 됨).
    """
    if isinstance(mode, str):
        mode = RegimeMode(mode)
    if mode is RegimeMode.OFF:
        return pd.Series(1.0, index=market.index, dtype=float)

    min_periods = max(20, int(ma_window * min_periods_ratio))
    ma = market.rolling(window=ma_window, min_periods=min_periods).mean()
    above = market > ma

    if mode is RegimeMode.BINARY:
        return above.astype(float)
    if mode is RegimeMode.HALF:
        return above.astype(float).where(above, 0.5)
    raise ValueError(f"unknown mode: {mode}")


def apply_regime_filter(
    weights: pd.DataFrame,
    exposure: pd.Series,
    *,
    fill_initial: float = 1.0,
) -> pd.DataFrame:
    """가중치 패널 × 일별 노출 비율.

    exposure가 비어있는(MA 워밍업 전) 구간은 fill_initial로 채움 (기본 1.0=정상 매매).
    """
    aligned = exposure.reindex(weights.index, method="ffill").fillna(fill_initial)
    return weights.mul(aligned, axis=0)


def filter_weights(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    ma_window: int = 200,
    mode: str | RegimeMode = RegimeMode.BINARY,
) -> tuple[pd.DataFrame, pd.Series]:
    """편의 함수: prices에서 proxy 계산 → exposure → weights에 적용.

    Returns:
        (필터된 weights, exposure 시리즈) — exposure는 디버깅·로깅용.
    """
    proxy = market_proxy(prices)
    exposure = regime_exposure(proxy, ma_window=ma_window, mode=mode)
    return apply_regime_filter(weights, exposure), exposure


def regime_state(
    prices: pd.DataFrame,
    *,
    ma_window: int = 200,
    as_of: pd.Timestamp | None = None,
) -> dict:
    """현재 (또는 지정 시점) 시장 상태 — 시그널 출력용.

    Returns dict with: market_value, ma_value, distance_pct, in_uptrend, recommendation
    """
    proxy = market_proxy(prices)
    ma = proxy.rolling(window=ma_window, min_periods=max(20, ma_window // 2)).mean()
    if as_of is None:
        as_of = proxy.index[-1]
    if as_of not in proxy.index:
        idx = proxy.index[proxy.index <= as_of]
        if len(idx) == 0:
            raise ValueError(f"as_of {as_of} 이전 데이터 없음")
        as_of = idx[-1]

    m_val = float(proxy.loc[as_of])
    ma_val = float(ma.loc[as_of])
    distance = (m_val / ma_val - 1.0) * 100 if ma_val > 0 else 0.0
    in_up = m_val > ma_val
    return {
        "as_of": str(as_of.date()),
        "market_value": m_val,
        "ma_value": ma_val,
        "distance_pct": distance,
        "in_uptrend": in_up,
        "recommendation": "풀투자 (시장 > MA200)" if in_up else "현금화 (시장 < MA200)",
    }
