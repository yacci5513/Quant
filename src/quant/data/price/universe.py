"""시점별 종목 유니버스 (point-in-time universe).

가드레일 §2 (생존편향) 강화:
- fdr.StockListing("KOSPI")은 현재 시점 상장종목만 → 과거 백테스트 시 생존편향
- 본 모듈은 매 리밸런싱 일자별로 "그 시점에 살아있던 KOSPI 시총 상위 N"을 추정

설계:
- 우리는 이미 fetch_krx로 종목별 일봉을 받아둠
- 시가총액(market cap) 직접 데이터는 없지만, **종가 × 발행주식수** 로 근사 가능
- 발행주식수는 시간에 따라 변하지만 대형주는 안정적 → 현재 발행주식수로 근사하면 적당히 정확

방식:
1. 현재 종목 마스터(fdr StockListing)에서 종목 + 발행주식수(Stocks) 가져오기
2. 각 일자 t에 대해: market_cap[t][ticker] = close[t] × stocks[ticker]
3. t 시점 KOSPI 200 ≈ 그날 시총 상위 200 (저장된 종목 풀에서)

한계:
- 신규 상장·상장폐지 종목은 우리 데이터에 없으면 누락 (완벽한 PIT은 KRX/Refinitiv 유료 데이터 필요)
- 발행주식수 변동(증자/감자) 반영 안 됨
- 그러나 단순 fdr.StockListing 한계 대비 큰 개선 (fixed roster 대신 시점별 상위 N 추출)
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant.common.logger import logger


def load_shares_outstanding() -> pd.Series:
    """현재 KOSPI 종목별 발행주식수 (Stocks).

    fdr.StockListing 기준이라 현 시점값이지만, 대형주는 시간에 안정적.
    """
    import FinanceDataReader as fdr  # noqa: N813

    listing = fdr.StockListing("KOSPI")
    if "Stocks" not in listing.columns:
        raise ValueError("KOSPI listing에 Stocks 컬럼 없음")
    s = listing.set_index("Code")["Stocks"].astype(float)
    return s


def market_cap_panel(
    prices: pd.DataFrame,
    shares: pd.Series | None = None,
) -> pd.DataFrame:
    """시점별 시가총액 패널.

    cap[t, ticker] = close[t, ticker] × shares[ticker]
    """
    if shares is None:
        shares = load_shares_outstanding()
    common = [t for t in prices.columns if t in shares.index]
    if not common:
        raise ValueError("prices columns ↔ shares index 교집합 없음")
    return prices[common] * shares.loc[common]


def universe_at(
    cap_panel: pd.DataFrame,
    as_of: date | pd.Timestamp,
    top_n: int = 200,
) -> list[str]:
    """as_of 시점 시가총액 상위 N 종목 코드."""
    ts = pd.Timestamp(as_of)
    if ts not in cap_panel.index:
        # 가장 가까운 직전 거래일
        prior = cap_panel.index[cap_panel.index <= ts]
        if len(prior) == 0:
            return []
        ts = prior[-1]
    row = cap_panel.loc[ts].dropna()
    return row.nlargest(top_n).index.tolist()


def filter_panel_by_universe(
    panel: pd.DataFrame,
    cap_panel: pd.DataFrame,
    as_of: date | pd.Timestamp,
    top_n: int = 200,
) -> pd.DataFrame:
    """as_of 시점 유니버스에 속한 종목들만 추출 (다른 종목 = NaN 마스킹)."""
    universe = universe_at(cap_panel, as_of, top_n)
    out = panel.copy()
    other_cols = [c for c in panel.columns if c not in universe]
    out.loc[:, other_cols] = float("nan")
    return out


def time_series_universe_mask(
    cap_panel: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    top_n: int = 200,
) -> pd.DataFrame:
    """리밸런싱 일자별 유니버스 마스크 (True = 그 시점 유니버스에 포함).

    리밸런싱 사이엔 forward-fill (해당 윈도우의 유니버스 유지).
    """
    rows = []
    for d in rebalance_dates:
        in_uni = pd.Series(False, index=cap_panel.columns)
        try:
            tickers = universe_at(cap_panel, d, top_n)
            in_uni.loc[tickers] = True
        except Exception as e:
            logger.warning(f"universe_at({d}) 실패: {e}")
        in_uni.name = pd.Timestamp(d)
        rows.append(in_uni)
    df = pd.DataFrame(rows)
    df.index = rebalance_dates
    # 전체 일자로 ffill
    full = df.reindex(cap_panel.index, method="ffill")
    return full.fillna(False).astype(bool)
