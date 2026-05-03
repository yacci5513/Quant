"""KRX(KOSPI/KOSDAQ) 일봉 데이터 수집.

pykrx로 KOSPI 200 구성종목의 일봉 OHLCV를 받아 종목별 Parquet으로 저장한다.
증분 업데이트 지원 — 기존 파일이 있으면 마지막 일자 다음 날부터만 가져옴.

저장 레이아웃:
    data/raw/prices/{ticker}.parquet      # 종목 1개당 1파일

CLI:
    quant data fetch-krx               # KOSPI 200 5년치
    quant data fetch-krx --years 1     # 1년치
    quant data fetch-krx --tickers 005930,000660
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pykrx import stock
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.common.config import get_settings
from quant.common.logger import logger

# pykrx는 KRX 사이트를 스크래핑 → 너무 빠르면 차단당할 수 있음
_REQUEST_INTERVAL_SEC = 0.2
_DATE_FMT = "%Y%m%d"
_KOSPI200_INDEX = "1028"  # KRX 인덱스 코드


# -----------------------------------------------------------------------------
# 종목 마스터
# -----------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def fetch_kospi200_tickers(as_of: date | None = None) -> list[str]:
    """특정 일자 기준 KOSPI 200 구성종목 리스트.

    as_of가 None이면 오늘 기준. 영업일이 아니면 직전 영업일을 자동 사용.
    """
    as_of = as_of or date.today()
    tickers = stock.get_index_portfolio_deposit_file(_KOSPI200_INDEX, as_of.strftime(_DATE_FMT))
    if not tickers:
        # 비영업일이면 빈 리스트 반환됨 → 직전 영업일로 재시도
        prev = as_of - timedelta(days=1)
        logger.warning(f"KOSPI 200 종목 조회 실패 ({as_of}), {prev}로 재시도")
        return fetch_kospi200_tickers(prev)
    logger.info(f"KOSPI 200 구성종목 {len(tickers)}개 (기준일 {as_of})")
    return tickers


# -----------------------------------------------------------------------------
# 일봉 OHLCV
# -----------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def fetch_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    """단일 종목 일봉. start/end 포함.

    반환 컬럼: open, high, low, close, volume, value, change_pct
    인덱스: pandas.DatetimeIndex (KST 자정)
    """
    df = stock.get_market_ohlcv_by_date(
        start.strftime(_DATE_FMT),
        end.strftime(_DATE_FMT),
        ticker,
    )
    if df.empty:
        return df
    df = df.rename(
        columns={
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
            "거래대금": "value",
            "등락률": "change_pct",
        }
    )
    df.index.name = "date"
    return df


# -----------------------------------------------------------------------------
# 저장/적재
# -----------------------------------------------------------------------------
def _ticker_path(ticker: str, base: Path | None = None) -> Path:
    base = base or (get_settings().data_dir / "raw" / "prices")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{ticker}.parquet"


def _last_stored_date(path: Path) -> date | None:
    """기존 parquet의 마지막 일자 (없으면 None)."""
    if not path.exists():
        return None
    existing = pd.read_parquet(path)
    if existing.empty:
        return None
    return existing.index.max().date()


def save_ohlcv(ticker: str, df: pd.DataFrame, base: Path | None = None) -> Path:
    """종목 OHLCV를 parquet에 저장. 기존 데이터가 있으면 머지(중복 제거)."""
    path = _ticker_path(ticker, base)
    if path.exists():
        existing = pd.read_parquet(path)
        merged = pd.concat([existing, df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = df.sort_index()
    merged.to_parquet(path, compression="snappy")
    return path


# -----------------------------------------------------------------------------
# 메인 진입점
# -----------------------------------------------------------------------------
def update_ticker(
    ticker: str,
    end: date,
    default_start: date,
    base: Path | None = None,
) -> int:
    """단일 종목 증분 업데이트. 추가된 행 수 반환.

    기존 데이터가 있으면 마지막 일자 다음 날부터, 없으면 default_start부터.
    """
    path = _ticker_path(ticker, base)
    last = _last_stored_date(path)
    start = (last + timedelta(days=1)) if last else default_start
    if start > end:
        return 0
    df = fetch_ohlcv(ticker, start, end)
    if df.empty:
        return 0
    save_ohlcv(ticker, df, base)
    return len(df)


def fetch_all(
    tickers: Iterable[str] | None = None,
    years: int = 5,
    end: date | None = None,
    base: Path | None = None,
) -> dict[str, int]:
    """KOSPI 200 (또는 지정 종목)의 일봉을 가져와 저장.

    Returns:
        ticker → 추가된 행 수
    """
    end = end or date.today()
    default_start = end - timedelta(days=365 * years)
    if tickers is None:
        tickers = fetch_kospi200_tickers(end)

    results: dict[str, int] = {}
    tickers_list = list(tickers)
    total = len(tickers_list)

    for i, ticker in enumerate(tickers_list, 1):
        try:
            added = update_ticker(ticker, end, default_start, base)
            results[ticker] = added
            logger.info(f"[{i}/{total}] {ticker}: +{added} rows")
        except Exception as e:
            logger.exception(f"[{i}/{total}] {ticker} 실패: {e}")
            results[ticker] = -1
        time.sleep(_REQUEST_INTERVAL_SEC)

    ok = sum(1 for n in results.values() if n >= 0)
    failed = sum(1 for n in results.values() if n < 0)
    logger.info(f"완료: 성공 {ok}, 실패 {failed}, 총 {total}")
    return results


def load_close_panel(
    tickers: Iterable[str] | None = None,
    base: Path | None = None,
) -> pd.DataFrame:
    """저장된 parquet들을 합쳐 종가 패널(wide format) 반환.

    인덱스: date, 컬럼: ticker, 값: close. 백테스트 입력용.
    """
    base_path = base or (get_settings().data_dir / "raw" / "prices")
    if tickers is None:
        tickers = sorted(p.stem for p in base_path.glob("*.parquet"))

    series_list = []
    for ticker in tickers:
        path = _ticker_path(ticker, base)
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["close"])
        series_list.append(df["close"].rename(ticker))

    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1).sort_index()
