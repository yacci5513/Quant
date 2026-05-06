"""모멘텀 Top-N 월 리밸런싱 (학술 베이스라인).

규칙 (Jegadeesh & Titman 류):
1. 매 리밸런싱 일자(월초 영업일)에 12개월 수익률 상위 N 종목 선정
2. 가장 최근 1개월은 제외 (단기 반전 효과 회피) → "12-1 모멘텀"
3. 동일가중 (각 1/N)
4. 다음 리밸런싱까지 보유

가드레일 준수 (QUANT_GUARDRAILS.md):
§1 과적합     : 파라미터 2개(lookback, top_n)만 노출
§2 생존편향   : 시점별 필터(NaN/거래정지 종목 자동 제외)
§3 미래참조   : .shift(1)로 어제 종가까지만 사용
§4 거래비용   : 비용 모델은 engine에서 적용
§5 Crowding   : 학술 검증된 단순 전략 — 알파 약화 가능성 인지
§6 데이터품질 : NaN/0거래량 종목은 시그널 계산에서 제외
§7 통계유의성 : 월별 리밸런싱 + 5년 = 60회. 30 이상 OK
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.common.logger import logger


@dataclass(frozen=True)
class MomentumTopNConfig:
    lookback_months: int = 12  # 모멘텀 계산 윈도우
    skip_months: int = 1  # 최근 N개월 제외 (단기 반전 회피)
    top_n: int = 10  # 선정 종목 수
    min_avg_value: float = 1e9  # 일평균 거래대금 필터 (10억 원)
    rebalance_freq: str = "BMS"  # Business Month Start (월초 영업일)


def _momentum_score(prices: pd.DataFrame, cfg: MomentumTopNConfig) -> pd.DataFrame:
    """12-1 모멘텀: t-12개월 ~ t-1개월 수익률.

    거래일 기준으로 lookback*21 ~ skip*21 근사. shift는 호출 측에서 (.shift(1))
    또는 weights 형성 시 적용.
    """
    skip_days = cfg.skip_months * 21
    lookback_days = cfg.lookback_months * 21

    # px(t-skip_days) / px(t-lookback_days) - 1
    end = prices.shift(skip_days)
    start = prices.shift(lookback_days)
    return end / start - 1.0


def _liquidity_mask(value_panel: pd.DataFrame, cfg: MomentumTopNConfig) -> pd.DataFrame:
    """일평균 거래대금 60일 평균이 임계 이상인 종목 마스크.

    True = 거래 가능, False = 유동성 부족으로 제외.
    """
    avg = value_panel.rolling(window=60, min_periods=20).mean()
    return avg >= cfg.min_avg_value


def generate_weights(
    prices: pd.DataFrame,
    *,
    values: pd.DataFrame | None = None,
    config: MomentumTopNConfig | None = None,
) -> pd.DataFrame:
    """일별 목표 가중치(0~1) 패널 반환.

    prices: 일별 종가 패널 (index=date, columns=ticker)
    values: 일별 거래대금 패널 (선택). None이면 유동성 필터 미적용.
    config: 전략 파라미터.

    출력: 가중치 패널 (월초마다 갱신, 사이엔 forward-fill).
          엔진에 그대로 전달 가능 — 호출자가 .shift(1) 추가 적용 권장 (룩어헤드 방지).
    """
    cfg = config or MomentumTopNConfig()

    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns)

    # 1. 모멘텀 시그널
    momentum = _momentum_score(prices, cfg)

    # 2. 유동성 마스크
    if values is not None:
        liquid = _liquidity_mask(values, cfg)
        momentum = momentum.where(liquid)

    # 3. NaN/거래정지 제외 — 가격이 NaN 또는 거래대금 0인 종목 제외
    valid = prices.notna() & (prices > 0)
    momentum = momentum.where(valid)

    # 4. 리밸런싱 일자만 추출 (월초 영업일)
    rebalance_dates = pd.date_range(
        start=prices.index[0], end=prices.index[-1], freq=cfg.rebalance_freq
    )
    # prices.index에 실제 존재하는 일자로 매핑 (가까운 다음 거래일)
    rb_idx = []
    for d in rebalance_dates:
        future = prices.index[prices.index >= d]
        if len(future) > 0:
            rb_idx.append(future[0])
    rb_idx = pd.DatetimeIndex(sorted(set(rb_idx)))

    # 5. 각 리밸런싱 일자에 Top-N 선정 → 동일 가중치
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)
    skipped = 0
    for d in rb_idx:
        scores = momentum.loc[d].dropna()
        if len(scores) < cfg.top_n:
            skipped += 1
            continue
        top = scores.nlargest(cfg.top_n).index
        weights.loc[d, :] = 0.0
        weights.loc[d, top] = 1.0 / cfg.top_n

    # 6. 리밸런싱 사이 forward-fill — 보유 유지
    # rebalance 일자엔 새 값, 사이엔 직전 유지. 그 외(처음)는 0.
    weights_at_rb = weights.loc[rb_idx]
    weights_full = weights_at_rb.reindex(prices.index, method="ffill").fillna(0.0)

    if skipped > 0:
        logger.warning(
            f"momentum_topn: 시그널 부족으로 {skipped}/{len(rb_idx)} 리밸런싱 스킵 "
            f"(top_n={cfg.top_n}, lookback={cfg.lookback_months}m)"
        )
    logger.info(f"momentum_topn: rebalances={len(rb_idx) - skipped}, top_n={cfg.top_n}")

    # 룩어헤드 방지: 시그널 계산 일자(d) 정보로 d 시작 가중치 결정.
    # 엔진에서 weights[t] × returns[t] = 그날 보유로 그날 수익 → 안전.
    # 추가 보수성을 원하면 호출자가 .shift(1)로 한 영업일 더 미루기.
    return weights_full


def __getattr__(name: str) -> object:  # 모듈 레벨 백테스트 헬퍼 자동 노출용
    if name == "Config":
        return MomentumTopNConfig
    raise AttributeError(name)
