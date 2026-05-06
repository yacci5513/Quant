"""KRX(KOSPI/KOSDAQ) 일봉 데이터 수집.

FinanceDataReader 기반 (pykrx 1.0.51은 KRX 웹 구조 변경으로 종목 마스터 함수 깨짐).
KOSPI 시가총액 상위 N으로 KOSPI 200을 근사.

저장 레이아웃:
    data/raw/prices/{ticker}.parquet      # 종목 1개당 1파일

CLI:
    quant data fetch-krx               # KOSPI 시총 상위 200개, 5년치
    quant data fetch-krx --years 1     # 1년치
    quant data fetch-krx --tickers 005930,000660

가드레일 §2 (생존 편향) 한계:
    StockListing은 현재 시점 KOSPI 종목만 반환 — 과거 시점 마스터 추적 불가.
    백테스트 시 인지하고 결과 해석 (Phase 1 베이스라인 수용).
    더 엄격하려면 매 리밸런싱 일자별 KRX 공시 OPEN API 호출 (Phase 후속).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import FinanceDataReader as fdr  # noqa: N813 (라이브러리 표준 alias)
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.common.config import get_settings
from quant.common.logger import logger

# fdr는 KRX/네이버 등 여러 소스를 사용 — 너무 빠르면 차단당할 수 있음
_REQUEST_INTERVAL_SEC = 0.1
_KOSPI_DEFAULT_TOP_N = 200


# -----------------------------------------------------------------------------
# 종목 마스터
# -----------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def fetch_kospi200_tickers(top_n: int = _KOSPI_DEFAULT_TOP_N) -> list[str]:
    """KOSPI 시총 상위 N개 종목 (KOSPI 200 근사).

    실제 KOSPI 200 인덱스 구성과 100% 일치하지 않지만(섹터 대표성·유동성 등 추가
    심사 있음), 백테스트 베이스라인으로는 충분. 시총 상위 200개의 약 90% 이상이
    실제 KOSPI 200과 겹친다.
    """
    listing = fdr.StockListing("KOSPI")
    if listing.empty or "Marcap" not in listing.columns:
        raise ValueError("FinanceDataReader StockListing(KOSPI) 결과 비정상")
    listing = listing.dropna(subset=["Marcap"])
    listing = listing.sort_values("Marcap", ascending=False)
    tickers = listing["Code"].head(top_n).tolist()
    logger.info(f"KOSPI 시총 상위 {len(tickers)}개 (KOSPI 200 근사)")
    return tickers


# -----------------------------------------------------------------------------
# 일봉 OHLCV
# -----------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def fetch_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    """단일 종목 일봉. start/end 포함.

    반환 컬럼: open, high, low, close, volume, value, change_pct
    인덱스: pandas.DatetimeIndex (이름='date')
    value = close * volume (거래대금 근사)
    """
    df = fdr.DataReader(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df.empty:
        return df
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Change": "change_pct",
        }
    )
    # fdr는 Amount(거래대금) 미제공 → close × volume 으로 근사
    df["value"] = df["close"] * df["volume"]
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume", "value", "change_pct"]]


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
    """KOSPI 시총 상위 N종목 (또는 지정 종목)의 일봉을 가져와 저장.

    Returns:
        ticker → 추가된 행 수 (실패 시 -1)
    """
    end = end or date.today()
    default_start = end - timedelta(days=365 * years)
    if tickers is None:
        tickers = fetch_kospi200_tickers()

    results: dict[str, int] = {}
    tickers_list = list(tickers)
    total = len(tickers_list)

    for i, ticker in enumerate(tickers_list, 1):
        try:
            added = update_ticker(ticker, end, default_start, base)
            results[ticker] = added
            if i % 20 == 0 or added > 0:
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
    return _load_panel("close", tickers, base)


def load_value_panel(
    tickers: Iterable[str] | None = None,
    base: Path | None = None,
) -> pd.DataFrame:
    """거래대금(value) 패널 (wide). 유동성 필터용."""
    return _load_panel("value", tickers, base)


def _load_panel(
    column: str,
    tickers: Iterable[str] | None,
    base: Path | None,
) -> pd.DataFrame:
    base_path = base or (get_settings().data_dir / "raw" / "prices")
    if tickers is None:
        tickers = sorted(p.stem for p in base_path.glob("*.parquet"))

    series_list = []
    for ticker in tickers:
        path = _ticker_path(ticker, base)
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=[column])
        series_list.append(df[column].rename(ticker))

    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1).sort_index()
