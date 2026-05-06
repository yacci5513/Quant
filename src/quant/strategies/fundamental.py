"""펀더멘털 시그널 전략 (Quality + Value 결합).

근거:
- Fama-French (1992): 저PBR이 장기 초과 수익 (Value)
- Asness, Frazzini, Pedersen (2014) "Quality Minus Junk": 고ROE 안정 알파 (Quality)
- Piotroski F-Score, Greenblatt Magic Formula 류 결합 시그널

규칙:
1. 매 리밸런싱 일자에 종목별 PER/PBR/ROE/영업이익률 계산
2. 각 지표 z-score → 점수 합산 (가중치 옵션)
3. 종합 점수 Top-N 선정 → 동일가중

가드레일:
§3 미래참조: 사업보고서 공시 후 90영업일 후부터 시그널 적용 (감사 정정 회피)
§6 데이터품질: NaN/0 분모는 제외, 음수 자본/순이익은 별도 처리
§7 통계유의성: 연 1회 사업보고서 기반 → 분기보고서 추가 확장 권장
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.common.logger import logger


@dataclass(frozen=True)
class FundamentalConfig:
    top_n: int = 10
    rebalance_freq: str = "BMS"
    publication_lag_days: int = 90  # 사업연도말 후 영업일
    min_avg_value: float = 1e9  # 일평균 거래대금 임계
    weight_value: float = 0.5  # 1/PER + 1/PBR 가중치
    weight_quality: float = 0.5  # ROE + 영업이익률 가중치
    require_positive_earnings: bool = True  # 적자 종목 제외


def _zscore_cross_sectional(s: pd.Series) -> pd.Series:
    """결측 제외 + 한 시점 횡단 z-score. 결측은 0(중립)."""
    valid = s.dropna()
    if len(valid) < 5:
        return pd.Series(0.0, index=s.index)
    z = (s - valid.mean()) / (valid.std(ddof=0) + 1e-9)
    return z.fillna(0.0)


def _financials_to_daily_panel(
    financials: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    publication_lag_days: int = 90,
) -> dict[str, pd.DataFrame]:
    """fetch_financials_panel(long) → 일별 패널 4종.

    bsns_year=Y의 사업보고서는 (Y+1년 4월 1일 + lag_days) 시점부터 적용.
    실제 공시일 평균 = 3월 말, 감사인 정정 위험 1개월 = 4월 말 적용 OK.
    publication_lag_days 90 = 사업연도말 후 90영업일 후 ≈ 5월 초.
    """
    out = {}
    for metric in ["revenue", "operating_income", "net_income", "equity", "assets"]:
        if metric not in financials.columns:
            continue
        # 종목 × 연도 wide
        wide = financials.pivot_table(
            index="bsns_year", columns="stock_code", values=metric, aggfunc="first"
        )
        # 각 연도의 적용 시작 일자 = 다음 해 1월 1일 + lag_days 영업일
        panel = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns, dtype=float)
        for year, row in wide.iterrows():
            apply_from = pd.Timestamp(f"{int(year) + 1}-01-01")
            future = prices.index[prices.index >= apply_from]
            if len(future) <= publication_lag_days:
                continue
            apply_idx = future[publication_lag_days:]
            for ticker in row.dropna().index:
                if ticker in panel.columns:
                    panel.loc[apply_idx, ticker] = row[ticker]
        out[metric] = panel
    return out


def _liquidity_mask(value_panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    avg = value_panel.rolling(window=60, min_periods=20).mean()
    return avg >= threshold


def generate_weights(
    prices: pd.DataFrame,
    financials: pd.DataFrame,
    *,
    shares: pd.Series,
    values: pd.DataFrame | None = None,
    config: FundamentalConfig | None = None,
) -> pd.DataFrame:
    """Quality+Value 종합 점수 Top-N 동일가중.

    Args:
        prices: 일별 종가 패널
        financials: long format 재무제표 (fetch_financials_panel 결과)
        shares: 종목별 발행주식수 (universe.load_shares_outstanding)
        values: 일별 거래대금 패널 (유동성 필터)

    Returns:
        가중치 패널 (index=date, columns=ticker, sum<=1.0).
    """
    cfg = config or FundamentalConfig()
    if prices.empty or financials.empty:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)

    panels = _financials_to_daily_panel(
        financials, prices, publication_lag_days=cfg.publication_lag_days
    )
    if "equity" not in panels or "net_income" not in panels:
        logger.warning("fundamental: equity 또는 net_income 패널 없음")
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # 시가총액 = close × shares
    common = [t for t in prices.columns if t in shares.index]
    market_cap = prices[common] * shares.loc[common]

    equity = panels["equity"].reindex(columns=prices.columns)
    net_income = panels["net_income"].reindex(columns=prices.columns)
    op_income = panels.get(
        "operating_income", pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    )
    revenue = panels.get(
        "revenue", pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    )

    # 펀더멘털 비율
    # PBR = market_cap / equity (낮을수록 매력)
    # PER = market_cap / net_income (낮을수록, 양수일 때만)
    # ROE = net_income / equity (높을수록)
    # 영업이익률 = op_income / revenue (높을수록)
    market_cap_aligned = market_cap.reindex_like(prices).ffill()

    pbr = market_cap_aligned / equity.where(equity > 0)
    per = market_cap_aligned / net_income.where(
        net_income > 0 if cfg.require_positive_earnings else net_income.notna()
    )
    roe = net_income / equity.where(equity > 0)
    op_margin = op_income / revenue.where(revenue > 0)

    # 리밸런싱 일자 추출
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
        # 각 비율의 z-score (낮을수록 좋은 건 부호 반전)
        z_pbr = _zscore_cross_sectional(-pbr.loc[d])  # 낮은 PBR 선호
        z_per = _zscore_cross_sectional(-per.loc[d])
        z_roe = _zscore_cross_sectional(roe.loc[d])
        z_op_margin = _zscore_cross_sectional(op_margin.loc[d])

        score = (
            cfg.weight_value * (z_pbr + z_per) / 2 + cfg.weight_quality * (z_roe + z_op_margin) / 2
        )

        # 유동성 필터
        if values is not None:
            liq = _liquidity_mask(values, cfg.min_avg_value).loc[d]
            score = score.where(liq, np.nan)

        # 양의 자본·이익만 (cfg.require_positive_earnings)
        if cfg.require_positive_earnings:
            valid = (equity.loc[d] > 0) & (net_income.loc[d] > 0)
            score = score.where(valid, np.nan)

        scores = score.dropna()
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
        logger.warning(f"fundamental: {skipped}/{len(rb_idx)} 리밸런싱 스킵")
    logger.info(f"fundamental: rebalances={len(rb_idx) - skipped}, top_n={cfg.top_n}")
    return weights_full
