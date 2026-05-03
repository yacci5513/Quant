"""Settings 모듈 검증."""

from __future__ import annotations

import pytest

from quant.common.config import KISMode, Settings, get_settings


def test_default_paper_mode_no_credentials_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """모의(paper) 모드는 자격증명 비어있어도 기동된다."""
    monkeypatch.delenv("KIS_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_APP_SECRET", raising=False)
    monkeypatch.setenv("KIS_MODE", "paper")

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.kis_mode is KISMode.PAPER
    assert s.kis_base_url == s.kis_base_url_paper


def test_live_mode_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """라이브 모드인데 자격증명 비어있으면 검증 실패."""
    monkeypatch.setenv("KIS_MODE", "live")
    monkeypatch.setenv("KIS_APP_KEY", "")
    monkeypatch.setenv("KIS_APP_SECRET", "")

    with pytest.raises(ValueError, match="KIS_APP_KEY"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_live_mode_with_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """라이브 모드 + 자격증명 채우면 통과 + base URL은 라이브 URL."""
    monkeypatch.setenv("KIS_MODE", "live")
    monkeypatch.setenv("KIS_APP_KEY", "test_key")
    monkeypatch.setenv("KIS_APP_SECRET", "test_secret")
    monkeypatch.setenv("KIS_ACCOUNT_NO", "12345678-01")

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.kis_mode is KISMode.LIVE
    assert s.kis_base_url == s.kis_base_url_live
    assert s.kis_app_key.get_secret_value() == "test_key"


def test_token_cache_path_separates_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """토큰 캐시 파일명에 모드가 포함되어야 한다(섞임 방지)."""
    monkeypatch.setenv("KIS_MODE", "paper")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert "paper" in s.kis_token_cache_path.name


def test_get_settings_is_cached() -> None:
    """get_settings는 동일 인스턴스를 반환한다."""
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
