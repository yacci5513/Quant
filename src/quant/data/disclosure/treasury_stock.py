"""자사주 매입 공시 fetcher.

근거: Ikenberry, Lakonishok, Vermaelen (1995) "Market Underreaction to Open Market Share Repurchases"
 — 자사주 매입 공시 후 4년간 평균 12% 초과 수익. 한국 시장도 검증.

세 가지 신호 분리 (강도 순):
1. 주식소각결정          — 매입 + 영구 소각 (가장 강한 알파, 자본 영구 감소)
2. 주요사항보고서(자기주식취득결정) — 매입 의사 결정 (사전 시그널)
3. 자기주식취득결과보고서  — 매입 완료 (사후 보고, 시그널 약함)

가드레일:
§3 미래참조: 공시 일자(rcept_dt) 다음 영업일부터 매수 시그널 적용
§5 Crowding: 한국 특화 알파 — 외국 헤지펀드는 한국어 공시 늦게 처리, 잔존
§6 데이터품질: report_nm 키워드 매칭 (DART의 pblntf_ty 코드는 비어있는 경우 多)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path

import pandas as pd

from quant.common.config import get_settings
from quant.common.logger import logger
from quant.data.disclosure.client import DartClient

# 자사주 공시 식별 키워드 (report_nm 매칭)
_TREASURY_KEYWORDS = {
    "buyback_decision": ["자기주식취득결정", "자사주취득결정"],
    "buyback_result": ["자기주식취득결과", "자사주취득결과"],
    "cancellation": ["주식소각결정", "자기주식소각"],
}


@dataclass(frozen=True)
class TreasuryEvent:
    stock_code: str
    corp_code: str
    corp_name: str
    rcept_dt: pd.Timestamp
    report_nm: str
    event_type: str  # buyback_decision | buyback_result | cancellation
    rcept_no: str  # DART 접수번호 (URL 구성에 사용 가능)


def _classify(report_nm: str) -> str | None:
    """report_nm을 자사주 이벤트 유형으로 분류. None = 자사주 공시 아님."""
    # 정정 공시는 [기재정정] 접두사 가짐 → 첫 발견을 신호로 보고 정정은 제외
    if report_nm.startswith("[기재정정]"):
        return None
    for event_type, kws in _TREASURY_KEYWORDS.items():
        for kw in kws:
            if kw in report_nm:
                return event_type
    return None


def fetch_treasury_events(
    *,
    stock_codes: list[str],
    start: date,
    end: date,
    client: DartClient | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """주어진 종목 리스트에 대한 자사주 이벤트 일자 panel.

    반환 DataFrame 컬럼:
        stock_code, corp_code, corp_name, rcept_dt, report_nm, event_type, rcept_no

    DART 검색 한도(페이지 100건/회) 회피 — 종목별로 호출.
    종목당 평균 1년 100건 미만이라 페이지 1로 충분 (희박한 종목 기준 보수적).
    """
    client = client or DartClient()
    cache_path = cache_path or (
        get_settings().data_dir / "raw" / "dart" / "treasury_events.parquet"
    )
    if cache_path.exists():
        logger.info(f"treasury_events 캐시 사용: {cache_path}")
        return pd.read_parquet(cache_path)

    corp_df = client.fetch_corp_codes()
    # stock_code 6자리 + 우선주(7자리, 끝 K/L 등) 모두 처리
    corp_df["stock_norm"] = corp_df["stock_code"].str.strip()
    code_to_corp = dict(zip(corp_df["stock_norm"], corp_df["corp_code"], strict=False))
    code_to_name = dict(zip(corp_df["stock_norm"], corp_df["corp_name"], strict=False))

    bgn_de = start.strftime("%Y%m%d")
    end_de = end.strftime("%Y%m%d")

    events: list[TreasuryEvent] = []
    for i, sc in enumerate(stock_codes, 1):
        corp_code = code_to_corp.get(sc)
        if not corp_code:
            logger.warning(f"[{i}/{len(stock_codes)}] {sc}: corp_code 매칭 실패 — 스킵")
            continue
        try:
            # DART API는 검색 기간이 길면 100건 초과 가능 — 분기 단위로 분할 호출
            quarter_starts = pd.date_range(start=bgn_de, end=end_de, freq="QS").strftime("%Y%m%d")
            quarter_starts = [*list(quarter_starts), end_de]
            for q_start, q_end in pairwise(quarter_starts):
                resp = client.search_disclosures(
                    corp_code=corp_code, bgn_de=q_start, end_de=q_end, page_count=100
                )
                for d in resp.list_:
                    rn = d.get("report_nm", "")
                    et = _classify(rn)
                    if et is None:
                        continue
                    events.append(
                        TreasuryEvent(
                            stock_code=sc,
                            corp_code=corp_code,
                            corp_name=code_to_name.get(sc, "?"),
                            rcept_dt=pd.to_datetime(d.get("rcept_dt"), format="%Y%m%d"),
                            report_nm=rn,
                            event_type=et,
                            rcept_no=str(d.get("rcept_no", "")),
                        )
                    )
            time.sleep(0.05)
            if i % 20 == 0:
                logger.info(f"[{i}/{len(stock_codes)}] 누적 이벤트 {len(events)}건")
        except Exception as e:
            logger.warning(f"[{i}/{len(stock_codes)}] {sc} ({corp_code}) 실패: {e}")

    df = pd.DataFrame([e.__dict__ for e in events])
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "stock_code",
                "corp_code",
                "corp_name",
                "rcept_dt",
                "report_nm",
                "event_type",
                "rcept_no",
            ]
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, compression="snappy")
    logger.info(f"treasury_events 저장: {len(df)}건 → {cache_path}")
    return df


def events_to_signal_panel(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
    decay_days: int = 60,
) -> pd.DataFrame:
    """이벤트 → 종목별 일별 시그널 점수 패널.

    weights:
        cancellation       기본 1.0  (가장 강한 알파)
        buyback_decision   기본 0.7  (사전 시그널)
        buyback_result     기본 0.3  (후행 보고)
    decay_days: 이벤트 후 N영업일에 걸쳐 점수 선형 감쇠. 그 후엔 0.

    Returns:
        index=prices.index, columns=prices.columns, values=점수 (0~1).
    """
    weights = weights or {"cancellation": 1.0, "buyback_decision": 0.7, "buyback_result": 0.3}
    panel = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)
    if events.empty:
        return panel

    for _, ev in events.iterrows():
        ticker = ev["stock_code"]
        if ticker not in panel.columns:
            continue
        base_w = weights.get(ev["event_type"], 0.0)
        if base_w <= 0:
            continue
        # 공시 일자 다음 영업일부터 시그널 적용 (룩어헤드 방지)
        rcept = pd.Timestamp(ev["rcept_dt"])
        future_dates = panel.index[panel.index > rcept]
        if len(future_dates) == 0:
            continue
        # 처음 decay_days 영업일에 선형 감쇠
        active = future_dates[:decay_days]
        n = len(active)
        if n == 0:
            continue
        decay = pd.Series([base_w * (1 - i / n) for i in range(n)], index=active, dtype=float)
        # 기존 점수와 max (중복 이벤트 시 더 강한 것 유지)
        panel.loc[active, ticker] = panel.loc[active, ticker].combine(decay, max).values
    return panel
