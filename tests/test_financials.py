"""DART 재무제표 fetcher 테스트 (네트워크 미사용)."""

from __future__ import annotations

from quant.data.disclosure.financials import _parse_amount, _parse_response_to_financials


def test_parse_amount_with_commas() -> None:
    assert _parse_amount("1,234,567") == 1234567.0


def test_parse_amount_negative() -> None:
    assert _parse_amount("-1,000") == -1000.0


def test_parse_amount_invalid() -> None:
    assert _parse_amount("") is None
    assert _parse_amount("-") is None
    assert _parse_amount(None) is None
    assert _parse_amount("abc") is None


def test_parse_response_with_consolidated() -> None:
    """fs_div='CFS' (연결) 우선 선택."""
    list_ = [
        {"fs_div": "CFS", "account_nm": "매출액", "thstrm_amount": "100,000"},
        {"fs_div": "CFS", "account_nm": "당기순이익", "thstrm_amount": "10,000"},
        {"fs_div": "OFS", "account_nm": "매출액", "thstrm_amount": "50,000"},
    ]
    f = _parse_response_to_financials(
        list_, stock_code="005930", corp_code="00126380", bsns_year=2024, reprt_code="11011"
    )
    assert f.revenue == 100_000
    assert f.net_income == 10_000


def test_parse_response_falls_back_to_separate() -> None:
    """fs_div='CFS' 없으면 OFS(별도) 사용."""
    list_ = [
        {"fs_div": "OFS", "account_nm": "매출액", "thstrm_amount": "50,000"},
        {"fs_div": "OFS", "account_nm": "영업이익", "thstrm_amount": "5,000"},
    ]
    f = _parse_response_to_financials(
        list_, stock_code="A", corp_code="X", bsns_year=2024, reprt_code="11011"
    )
    assert f.revenue == 50_000
    assert f.operating_income == 5_000


def test_parse_response_handles_synonyms() -> None:
    """'당기순이익(손실)' 같은 변형명도 매핑."""
    list_ = [
        {"fs_div": "CFS", "account_nm": "당기순이익(손실)", "thstrm_amount": "-1,000"},
        {"fs_div": "CFS", "account_nm": "자본총계", "thstrm_amount": "100,000"},
    ]
    f = _parse_response_to_financials(
        list_, stock_code="A", corp_code="X", bsns_year=2024, reprt_code="11011"
    )
    assert f.net_income == -1_000
    assert f.equity == 100_000


def test_parse_response_missing_fields_are_none() -> None:
    """일부 계정 누락 시 그 필드만 None."""
    list_ = [
        {"fs_div": "CFS", "account_nm": "매출액", "thstrm_amount": "100,000"},
    ]
    f = _parse_response_to_financials(
        list_, stock_code="A", corp_code="X", bsns_year=2024, reprt_code="11011"
    )
    assert f.revenue == 100_000
    assert f.operating_income is None
    assert f.equity is None
