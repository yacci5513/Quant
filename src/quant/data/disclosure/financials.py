"""DART 재무제표 fetcher (사업보고서 기반).

근거 (펀더멘털 알파):
- Fama-French (1992): PBR 낮은 종목이 장기 초과 수익 (Value)
- Asness, Frazzini, Pedersen (2014): ROE 높은 종목이 안정적 알파 (Quality)
- Piotroski (2000): F-Score (수익성·재무·운영효율) 결합 시 강한 시그널

핵심 계정 (DART fnlttSinglAcnt 응답):
- 매출액 (sales)
- 영업이익 (operating_income)
- 당기순이익 (net_income)
- 자본총계 (equity)
- 자산총계 (assets)
- 부채총계 (liabilities)

가드레일:
§3 미래참조: 사업보고서 공시 후 90영업일 후부터 시그널 적용 (감사인 의견·정정 회피)
§6 데이터품질: 결측치는 forward-fill 금지 — 결측 그대로 두고 시그널 계산 시 제외
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant.common.config import get_settings
from quant.common.logger import logger
from quant.data.disclosure.client import DartClient

# DART 응답 계정명 → 표준 키 매핑 (한국어 계정과목명)
_ACCOUNT_MAP = {
    "매출액": "revenue",
    "수익(매출액)": "revenue",
    "영업수익": "revenue",
    "영업이익": "operating_income",
    "영업이익(손실)": "operating_income",
    "당기순이익": "net_income",
    "당기순이익(손실)": "net_income",
    "당기순손익": "net_income",
    "자본총계": "equity",
    "자본": "equity",
    "자산총계": "assets",
    "부채총계": "liabilities",
}

# 보고서 코드
_REPORT_CODES = {
    "annual": "11011",  # 사업보고서 (연간)
    "q3": "11014",  # 3분기보고서
    "semi": "11012",  # 반기보고서
    "q1": "11013",  # 1분기보고서
}


@dataclass(frozen=True)
class Financials:
    stock_code: str
    corp_code: str
    bsns_year: int
    reprt_code: str
    revenue: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    equity: float | None = None
    assets: float | None = None
    liabilities: float | None = None


def _parse_amount(s: str | None) -> float | None:
    """DART는 amount를 콤마 포함 문자열로 줌 ("1,234,567")."""
    if not s or s == "-":
        return None
    try:
        return float(str(s).replace(",", "").replace(" ", ""))
    except (ValueError, TypeError):
        return None


def _parse_response_to_financials(
    list_: list[dict],
    *,
    stock_code: str,
    corp_code: str,
    bsns_year: int,
    reprt_code: str,
) -> Financials:
    """fnlttSinglAcnt 응답 list → Financials 단위.

    DART 응답은 계정과목별 row. account_nm으로 매핑.
    fs_div='CFS' (연결재무제표) 우선, 없으면 'OFS' (별도재무제표).
    """
    cfs = [r for r in list_ if r.get("fs_div") == "CFS"]
    chosen = cfs if cfs else list_

    values: dict[str, float] = {}
    for row in chosen:
        nm = row.get("account_nm", "").strip()
        std_key = _ACCOUNT_MAP.get(nm)
        if std_key and std_key not in values:
            amt = _parse_amount(row.get("thstrm_amount"))
            if amt is not None:
                values[std_key] = amt

    return Financials(
        stock_code=stock_code,
        corp_code=corp_code,
        bsns_year=bsns_year,
        reprt_code=reprt_code,
        **values,
    )


def fetch_financials_panel(
    *,
    stock_codes: list[str],
    years: list[int],
    reprt: str = "annual",
    client: DartClient | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """KOSPI 종목별 × 연도별 재무제표 패널.

    Args:
        stock_codes: 6자리 stock_code 리스트
        years: 사업연도 리스트 (예: [2020, 2021, 2022, 2023, 2024, 2025])
        reprt: 'annual'(사업) | 'q3' | 'semi' | 'q1'

    반환: long format DataFrame
        stock_code, corp_code, bsns_year, reprt_code,
        revenue, operating_income, net_income, equity, assets, liabilities
    """
    client = client or DartClient()
    cache_path = cache_path or (
        get_settings().data_dir / "raw" / "dart" / f"financials_{reprt}.parquet"
    )
    if cache_path.exists():
        logger.info(f"financials 캐시 사용: {cache_path}")
        return pd.read_parquet(cache_path)

    reprt_code = _REPORT_CODES[reprt]
    corp_df = client.fetch_corp_codes()
    code_to_corp = dict(zip(corp_df["stock_code"].str.strip(), corp_df["corp_code"], strict=False))

    rows: list[Financials] = []
    total = len(stock_codes) * len(years)
    done = 0
    for sc in stock_codes:
        corp_code = code_to_corp.get(sc)
        if not corp_code:
            done += len(years)
            continue
        for year in years:
            try:
                resp = client.fetch_financial_statements(
                    corp_code=corp_code, bsns_year=str(year), reprt_code=reprt_code
                )
                if resp.list_:
                    f = _parse_response_to_financials(
                        resp.list_,
                        stock_code=sc,
                        corp_code=corp_code,
                        bsns_year=year,
                        reprt_code=reprt_code,
                    )
                    rows.append(f)
                time.sleep(0.05)
            except Exception as e:
                logger.warning(f"{sc} {year}: {e}")
            done += 1
            if done % 100 == 0:
                logger.info(f"[{done}/{total}] 누적 {len(rows)}건")

    df = pd.DataFrame([r.__dict__ for r in rows])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, compression="snappy")
    logger.info(f"financials 저장: {len(df)}건 → {cache_path}")
    return df


def compute_ratios(
    financials: pd.DataFrame,
    *,
    market_cap: pd.DataFrame,
    publication_lag_days: int = 90,
) -> pd.DataFrame:
    """재무제표 + 시가총액 → 일별 PER/PBR/ROE 패널.

    Args:
        financials: fetch_financials_panel 결과 (long format)
        market_cap: 일별 시가총액 패널 (index=date, columns=stock_code)
        publication_lag_days: 사업연도말 후 N영업일 후부터 시그널 적용 (감사·정정 회피)

    Returns:
        dict of DataFrames: {"per": ..., "pbr": ..., "roe": ...}
        각 DataFrame: index=date, columns=stock_code
    """
    raise NotImplementedError("compute_ratios는 다음 단계에서 구현 — 펀더멘털 시그널 모듈에서 처리")
