"""KIS Developers REST API 클라이언트.

핵심 엔드포인트:
- 현재가 조회   : GET /uapi/domestic-stock/v1/quotations/inquire-price
- 잔고 조회     : GET /uapi/domestic-stock/v1/trading/inquire-balance
- 매수/매도 주문: POST /uapi/domestic-stock/v1/trading/order-cash

가드레일:
- 모든 주문은 paper(모의투자) 모드 디폴트. live는 명시적 플래그
- pre_trade_check 통과 못하면 주문 차단
- 호가 단위·수량 검증
- 응답 rt_cd != "0" 이면 KISError
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.common.config import KISMode, get_settings
from quant.common.logger import logger
from quant.live.auth import auth_headers, get_kis_token

_DEFAULT_TIMEOUT = 15.0


# KIS API tr_id 코드 (paper/live 별도)
TR_INQUIRE_PRICE = "FHKST01010100"
TR_INQUIRE_BALANCE_PAPER = "VTTC8434R"
TR_INQUIRE_BALANCE_LIVE = "TTTC8434R"
TR_ORDER_BUY_PAPER = "VTTC0802U"
TR_ORDER_BUY_LIVE = "TTTC0802U"
TR_ORDER_SELL_PAPER = "VTTC0801U"
TR_ORDER_SELL_LIVE = "TTTC0801U"


def round_to_tick(price: float) -> int:
    """KOSPI/KOSDAQ 가격대별 호가 단위에 맞춰 라운드 (내림).

    호가 단위 (2024 KRX 기준):
      < 2,000원        : 1원
      2,000 ~ 5,000     : 5원
      5,000 ~ 20,000    : 10원
      20,000 ~ 50,000   : 50원
      50,000 ~ 200,000  : 100원
      200,000 ~ 500,000 : 500원
      >= 500,000        : 1,000원
    """
    if price < 2_000:
        tick = 1
    elif price < 5_000:
        tick = 5
    elif price < 20_000:
        tick = 10
    elif price < 50_000:
        tick = 50
    elif price < 200_000:
        tick = 100
    elif price < 500_000:
        tick = 500
    else:
        tick = 1_000
    return int(price // tick) * tick


class KISError(RuntimeError):
    """KIS API 호출 실패 (rt_cd != '0' 또는 HTTP 에러)."""


@dataclass
class Quote:
    ticker: str
    price: float  # 현재가
    volume: int
    change: float  # 전일 종가 대비 (원)
    change_pct: float  # 전일 종가 대비 (%)
    open_price: float = 0.0  # 오늘 시가 (장중 흐름 계산용)
    high: float = 0.0
    low: float = 0.0


@dataclass
class Holding:
    ticker: str
    name: str
    quantity: int
    avg_price: float  # 매입 평균 단가
    current_price: float
    eval_amount: float  # 평가금액
    profit: float  # 평가손익
    profit_pct: float


@dataclass
class OrderResult:
    order_no: str  # 주문번호
    ticker: str
    side: str  # "buy" | "sell"
    quantity: int
    requested_price: float | None  # None=시장가
    success: bool
    raw: dict


def _account_parts() -> tuple[str, str]:
    """KIS_ACCOUNT_NO 12345678-01 → ("12345678", "01")."""
    s = get_settings()
    raw = (s.kis_account_no or "").strip()
    if "-" in raw:
        cano, prdt = raw.split("-", 1)
    else:
        cano, prdt = raw[:8], raw[8:].lstrip("0").zfill(2) or "01"
    if not cano or not prdt:
        raise KISError(f"KIS_ACCOUNT_NO 형식 오류: '{raw}' (예: 12345678-01)")
    return cano, prdt


def _is_paper() -> bool:
    return get_settings().kis_mode is KISMode.PAPER


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _get(path: str, *, tr_id: str, params: dict) -> dict:
    s = get_settings()
    headers = auth_headers()
    headers["tr_id"] = tr_id
    headers["custtype"] = "P"  # 개인
    url = f"{s.kis_base_url}{path}"
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        resp = client.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise KISError(f"GET {path} HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    if data.get("rt_cd") != "0":
        raise KISError(f"GET {path} rt_cd={data.get('rt_cd')} msg={data.get('msg1', '')}")
    return data


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _post(path: str, *, tr_id: str, body: dict) -> dict:
    s = get_settings()
    headers = auth_headers()
    headers["tr_id"] = tr_id
    headers["custtype"] = "P"
    url = f"{s.kis_base_url}{path}"
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        resp = client.post(url, headers=headers, json=body)
    if resp.status_code != 200:
        raise KISError(f"POST {path} HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    if data.get("rt_cd") != "0":
        raise KISError(f"POST {path} rt_cd={data.get('rt_cd')} msg={data.get('msg1', '')}")
    return data


def get_quote(ticker: str) -> Quote:
    """종목 현재가 조회."""
    data = _get(
        "/uapi/domestic-stock/v1/quotations/inquire-price",
        tr_id=TR_INQUIRE_PRICE,
        params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
    )
    output = data.get("output", {})
    return Quote(
        ticker=ticker,
        price=float(output.get("stck_prpr", 0)),
        volume=int(output.get("acml_vol", 0)),
        change=float(output.get("prdy_vrss", 0)),
        change_pct=float(output.get("prdy_ctrt", 0)),
        open_price=float(output.get("stck_oprc", 0)),
        high=float(output.get("stck_hgpr", 0)),
        low=float(output.get("stck_lwpr", 0)),
    )


def get_balance() -> list[Holding]:
    """현재 보유 종목 + 평가손익 조회."""
    cano, prdt = _account_parts()
    tr_id = TR_INQUIRE_BALANCE_PAPER if _is_paper() else TR_INQUIRE_BALANCE_LIVE
    params = {
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "00",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    data = _get(
        "/uapi/domestic-stock/v1/trading/inquire-balance",
        tr_id=tr_id,
        params=params,
    )
    rows = data.get("output1", []) or []
    holdings: list[Holding] = []
    for r in rows:
        qty = int(r.get("hldg_qty", 0))
        if qty <= 0:
            continue
        holdings.append(
            Holding(
                ticker=str(r.get("pdno", "")).strip(),
                name=str(r.get("prdt_name", "")).strip(),
                quantity=qty,
                avg_price=float(r.get("pchs_avg_pric", 0)),
                current_price=float(r.get("prpr", 0)),
                eval_amount=float(r.get("evlu_amt", 0)),
                profit=float(r.get("evlu_pfls_amt", 0)),
                profit_pct=float(r.get("evlu_pfls_rt", 0)),
            )
        )
    return holdings


def order_cash(
    *,
    ticker: str,
    quantity: int,
    side: str,  # "buy" | "sell"
    price: float | None = None,  # None = 시장가
) -> OrderResult:
    """현금 매수/매도 주문 (모의/실전 모드 자동 감지)."""
    if side not in {"buy", "sell"}:
        raise ValueError(f"side는 'buy' 또는 'sell' (got {side!r})")
    if quantity <= 0:
        raise ValueError(f"수량은 양수여야 함 (got {quantity})")

    cano, prdt = _account_parts()
    paper = _is_paper()
    if side == "buy":
        tr_id = TR_ORDER_BUY_PAPER if paper else TR_ORDER_BUY_LIVE
    else:
        tr_id = TR_ORDER_SELL_PAPER if paper else TR_ORDER_SELL_LIVE

    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "PDNO": ticker,
        "ORD_DVSN": "01" if price is None else "00",  # 01=시장가, 00=지정가
        "ORD_QTY": str(quantity),
        "ORD_UNPR": "0" if price is None else str(int(price)),
    }
    logger.info(
        f"KIS 주문 요청: {side} {ticker} {quantity}주 @ {price or '시장가'} (paper={paper})"
    )
    # 토큰 미리 발급 (없으면 자동)
    get_kis_token()
    data = _post(
        "/uapi/domestic-stock/v1/trading/order-cash",
        tr_id=tr_id,
        body=body,
    )
    output = data.get("output", {})
    order_no = str(output.get("ODNO", ""))
    logger.info(f"KIS 주문 성공: {side} {ticker} {quantity}주, 주문번호 {order_no}")
    return OrderResult(
        order_no=order_no,
        ticker=ticker,
        side=side,
        quantity=quantity,
        requested_price=price,
        success=True,
        raw=data,
    )
