"""KIS Auth 모듈 테스트 (네트워크 미사용)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from quant.live.auth import (
    KISAuthError,
    TokenCache,
    _load_cache,
    _save_cache,
)


def test_token_cache_valid_when_fresh() -> None:
    """방금 발급한 토큰은 유효."""
    cache = TokenCache(
        access_token="abc",
        token_type="Bearer",
        issued_at=time.time(),
        expires_in=86400,
    )
    assert cache.is_valid
    assert cache.remaining_seconds > 80000


def test_token_cache_invalid_near_expiry() -> None:
    """만료 30분 전이면 유효하지 않음 (1시간 안전 마진)."""
    cache = TokenCache(
        access_token="abc",
        token_type="Bearer",
        issued_at=time.time() - 86400 + 1800,  # 30분 남음
        expires_in=86400,
    )
    assert not cache.is_valid


def test_token_cache_invalid_expired() -> None:
    """만료된 토큰."""
    cache = TokenCache(
        access_token="abc",
        token_type="Bearer",
        issued_at=time.time() - 100000,
        expires_in=86400,
    )
    assert not cache.is_valid
    assert cache.remaining_seconds == 0


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "token.json"
    cache = TokenCache(
        access_token="my_token_xyz",
        token_type="Bearer",
        issued_at=12345.6,
        expires_in=86400,
    )
    _save_cache(cache_path, cache)
    loaded = _load_cache(cache_path)
    assert loaded is not None
    assert loaded.access_token == "my_token_xyz"
    assert loaded.issued_at == 12345.6


def test_load_returns_none_when_missing(tmp_path: Path) -> None:
    cache_path = tmp_path / "no_such.json"
    assert _load_cache(cache_path) is None


def test_load_returns_none_when_corrupt(tmp_path: Path) -> None:
    cache_path = tmp_path / "corrupt.json"
    cache_path.write_text("not valid json {{{")
    assert _load_cache(cache_path) is None


def test_save_sets_restrictive_perms(tmp_path: Path) -> None:
    """토큰 파일은 600 권한이어야 함 (소유자만 읽기)."""
    cache_path = tmp_path / "token.json"
    cache = TokenCache(
        access_token="abc", token_type="Bearer", issued_at=time.time(), expires_in=86400
    )
    _save_cache(cache_path, cache)
    mode = cache_path.stat().st_mode & 0o777
    assert mode == 0o600


def test_token_cache_dict_roundtrip() -> None:
    cache = TokenCache(access_token="abc", token_type="Bearer", issued_at=12345.6, expires_in=86400)
    d = cache.to_dict()
    restored = TokenCache.from_dict(d)
    assert restored.access_token == cache.access_token
    assert restored.issued_at == cache.issued_at


def test_kis_auth_error_is_runtime_error() -> None:
    """KISAuthError는 RuntimeError 서브클래스."""
    assert issubclass(KISAuthError, RuntimeError)


def test_load_cache_handles_missing_keys(tmp_path: Path) -> None:
    """필수 키 누락된 JSON → None."""
    cache_path = tmp_path / "incomplete.json"
    cache_path.write_text(json.dumps({"access_token": "abc"}))  # issued_at 누락
    assert _load_cache(cache_path) is None
