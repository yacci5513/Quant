"""KIS REST 클라이언트 테스트 (네트워크 미사용)."""

from __future__ import annotations

import pytest

from quant.live.client import KISError, _account_parts


def test_account_parts_with_dash(monkeypatch) -> None:
    """12345678-01 → ('12345678', '01')."""
    from quant.common import config as cfg

    cfg.get_settings.cache_clear()
    monkeypatch.setenv("KIS_ACCOUNT_NO", "12345678-01")
    cano, prdt = _account_parts()
    cfg.get_settings.cache_clear()
    assert cano == "12345678"
    assert prdt == "01"


def test_account_parts_without_dash(monkeypatch) -> None:
    """1234567801 → ('12345678', '01') (자동 분리)."""
    from quant.common import config as cfg

    cfg.get_settings.cache_clear()
    monkeypatch.setenv("KIS_ACCOUNT_NO", "1234567801")
    cano, prdt = _account_parts()
    cfg.get_settings.cache_clear()
    assert cano == "12345678"
    assert prdt == "01"


def test_account_parts_empty_raises(monkeypatch) -> None:
    """paper 모드 + 빈 계좌번호 → KISError (live 모드는 model_validator가 먼저 차단)."""
    from quant.common import config as cfg

    cfg.get_settings.cache_clear()
    monkeypatch.setenv("KIS_MODE", "paper")
    monkeypatch.setenv("KIS_ACCOUNT_NO", "")
    with pytest.raises(KISError, match="형식 오류"):
        _account_parts()
    cfg.get_settings.cache_clear()


def test_kis_error_is_runtime() -> None:
    assert issubclass(KISError, RuntimeError)


def test_quote_dataclass_fields() -> None:
    from quant.live.client import Quote

    q = Quote(ticker="005930", price=87800, volume=1000000, change=200, change_pct=0.23)
    assert q.ticker == "005930"
    assert q.price == 87800


def test_holding_dataclass_fields() -> None:
    from quant.live.client import Holding

    h = Holding(
        ticker="005930",
        name="삼성전자",
        quantity=10,
        avg_price=85000,
        current_price=87800,
        eval_amount=878000,
        profit=28000,
        profit_pct=3.29,
    )
    assert h.quantity == 10
    assert abs(h.profit_pct - 3.29) < 1e-6


def test_order_result_dataclass() -> None:
    from quant.live.client import OrderResult

    r = OrderResult(
        order_no="ABC123",
        ticker="005930",
        side="buy",
        quantity=10,
        requested_price=None,
        success=True,
        raw={},
    )
    assert r.success
    assert r.side == "buy"
    assert r.requested_price is None  # 시장가
