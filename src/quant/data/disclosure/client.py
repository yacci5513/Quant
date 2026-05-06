"""OpenDART (전자공시) API 클라이언트.

기본 정책:
- 모든 응답은 디스크 캐시 (data/raw/dart/cache/) — 동일 query 재호출 시 디스크 읽기
- rate limit: 보수적으로 0.1초/req (분당 600건 이하)
- API 키 누락 시 명확한 에러
- 응답 status가 "000"(정상)이 아니면 예외

주요 엔드포인트:
- list.json:           공시 검색 (corp_code, 기간, 보고서 유형)
- company.json:        회사 개요
- corpCode.xml:        전체 corp_code 매핑 (zip 압축, ~7 MB)
- fnlttSinglAcnt.json: 단일회사 주요계정 (재무제표)
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.common.config import get_settings
from quant.common.logger import logger

_BASE_URL = "https://opendart.fss.or.kr/api"
_DEFAULT_TIMEOUT = 30.0
_RATE_LIMIT_SEC = 0.1


class DartError(RuntimeError):
    """DART API 응답 에러 (status code != '000')."""


@dataclass(frozen=True)
class DartResponse:
    """DART API 응답 단위. status='000'이면 list 사용 가능."""

    status: str
    message: str
    list_: list[dict]


class DartClient:
    """OpenDART API 호출 + 디스크 캐시.

    Args:
        api_key: 인증키. 없으면 Settings.dart_api_key 사용.
        cache_dir: 응답 캐시 디렉토리. None이면 data/raw/dart/cache.
    """

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None) -> None:
        if api_key is None:
            settings = get_settings()
            api_key = settings.dart_api_key.get_secret_value()
        if not api_key:
            raise ValueError(
                "DART API 키가 설정되지 않음. .env에 DART_API_KEY 또는 생성자 인자로 전달"
            )
        self._api_key = api_key
        self._cache_dir = cache_dir or (get_settings().data_dir / "raw" / "dart" / "cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, endpoint: str, params: dict) -> Path:
        """endpoint + 정렬된 params로 결정적 캐시 경로."""
        # api_key는 캐시 키에서 제외 (키 변경에도 캐시 재사용)
        params_clean = {k: v for k, v in params.items() if k != "crtfc_key"}
        h = hashlib.sha1(
            json.dumps([endpoint, sorted(params_clean.items())], default=str).encode()
        ).hexdigest()[:16]
        return self._cache_dir / f"{endpoint}_{h}.json"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _get_json(self, endpoint: str, params: dict) -> dict:
        """JSON 엔드포인트 호출 + 디스크 캐시. status 검증."""
        cache = self._cache_path(endpoint, params)
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))

        params_with_key = {**params, "crtfc_key": self._api_key}
        url = f"{_BASE_URL}/{endpoint}.json"
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            resp = client.get(url, params=params_with_key)
            resp.raise_for_status()
            data = resp.json()

        time.sleep(_RATE_LIMIT_SEC)

        # status 검증 — '000'(정상), '013'(자료 없음, 정상으로 처리), 그 외 에러
        status = data.get("status", "")
        if status not in {"000", "013"}:
            raise DartError(f"{endpoint}: status={status}, message={data.get('message')}")

        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def search_disclosures(
        self,
        *,
        corp_code: str | None = None,
        bgn_de: str | None = None,
        end_de: str | None = None,
        pblntf_ty: str | None = None,
        page_no: int = 1,
        page_count: int = 100,
    ) -> DartResponse:
        """공시 검색 (list.json).

        Args:
            corp_code: 8자리 회사 고유번호. None이면 전체 회사
            bgn_de: 검색 시작일 (YYYYMMDD)
            end_de: 검색 종료일 (YYYYMMDD)
            pblntf_ty: 공시 유형 (A=정기공시, B=주요사항보고, J=자산양수도 등)
            page_no/page_count: 페이지네이션
        """
        params = {
            k: v
            for k, v in {
                "corp_code": corp_code,
                "bgn_de": bgn_de,
                "end_de": end_de,
                "pblntf_ty": pblntf_ty,
                "page_no": page_no,
                "page_count": page_count,
            }.items()
            if v is not None
        }
        data = self._get_json("list", params)
        return DartResponse(
            status=data.get("status", ""),
            message=data.get("message", ""),
            list_=data.get("list", []) or [],
        )

    def fetch_corp_codes(self) -> pd.DataFrame:
        """전체 corp_code ↔ stock_code 매핑 (corpCode.xml.zip).

        반환 DataFrame: corp_code, corp_name, stock_code, modify_date.
        한 번 받아 parquet로 영속화 권장.
        """
        cache = self._cache_dir.parent / "corp_codes.parquet"
        if cache.exists():
            logger.info(f"corp_codes 캐시 사용: {cache}")
            return pd.read_parquet(cache)

        url = f"{_BASE_URL}/corpCode.xml"
        params = {"crtfc_key": self._api_key}
        with httpx.Client(timeout=120.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            content = resp.content

        # zip → xml → DataFrame
        with zipfile.ZipFile(io.BytesIO(content)) as zf, zf.open("CORPCODE.xml") as f:
            df = pd.read_xml(f)

        # 컬럼 정리 — corp_code, corp_name, stock_code, modify_date 표준화
        df = df.rename(
            columns={
                "corp_code": "corp_code",
                "corp_name": "corp_name",
                "stock_code": "stock_code",
                "modify_date": "modify_date",
            }
        )
        df["corp_code"] = df["corp_code"].astype(str).str.zfill(8)
        df["stock_code"] = df["stock_code"].fillna("").astype(str).str.strip()
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, compression="snappy")
        logger.info(f"corp_codes 다운로드: {len(df):,}건 → {cache}")
        return df

    def fetch_financial_statements(
        self, *, corp_code: str, bsns_year: str, reprt_code: str = "11011"
    ) -> DartResponse:
        """단일회사 주요계정 (fnlttSinglAcnt).

        Args:
            corp_code: 8자리 회사 고유번호
            bsns_year: 사업연도 (YYYY)
            reprt_code: 보고서 코드 (11011=사업보고서, 11012=반기, 11013=1분기, 11014=3분기)

        Returns:
            list_: 계정과목별 row [{account_nm, thstrm_amount, frmtrm_amount, ...}]
        """
        params = {"corp_code": corp_code, "bsns_year": bsns_year, "reprt_code": reprt_code}
        data = self._get_json("fnlttSinglAcnt", params)
        return DartResponse(
            status=data.get("status", ""),
            message=data.get("message", ""),
            list_=data.get("list", []) or [],
        )
