"""KIS Developers OAuth 토큰 매니저.

KIS API:
- 액세스 토큰 24시간 유효
- 일일 발급 한도 존재 → 디스크 캐시 필수 (kis_token_cache_path)
- 모의/실전 키 분리, 토큰도 분리

토큰 발급 엔드포인트:
- POST /oauth2/tokenP
- body: {"grant_type": "client_credentials", "appkey": ..., "appsecret": ...}
- response: {"access_token": "...", "token_type": "Bearer", "expires_in": 86400}
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.common.config import KISMode, get_settings
from quant.common.logger import logger


@dataclass
class TokenCache:
    """디스크에 저장되는 토큰 정보."""

    access_token: str
    token_type: str
    issued_at: float  # Unix timestamp (seconds)
    expires_in: int  # seconds (default 86400 = 24h)

    @property
    def expires_at(self) -> float:
        return self.issued_at + self.expires_in

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - time.time())

    @property
    def is_valid(self) -> bool:
        # 만료 1시간 전부터 무효 처리 (안전 마진)
        return self.remaining_seconds > 3600

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "issued_at": self.issued_at,
            "expires_in": self.expires_in,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TokenCache:
        return cls(
            access_token=d["access_token"],
            token_type=d.get("token_type", "Bearer"),
            issued_at=d["issued_at"],
            expires_in=d.get("expires_in", 86400),
        )


class KISAuthError(RuntimeError):
    """KIS 인증 실패."""


def _load_cache(path: Path) -> TokenCache | None:
    if not path.exists():
        return None
    try:
        return TokenCache.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"토큰 캐시 손상 ({path}): {e} — 무시하고 재발급")
        return None


def _save_cache(path: Path, cache: TokenCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache.to_dict()), encoding="utf-8")
    # 권한 600 (소유자만 읽기) — 토큰 보호
    with contextlib.suppress(OSError):
        path.chmod(0o600)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _request_new_token(
    *, base_url: str, app_key: str, app_secret: str, timeout: float = 15.0
) -> TokenCache:
    """KIS 토큰 발급 API 호출."""
    url = f"{base_url}/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=body)
        if resp.status_code != 200:
            raise KISAuthError(f"토큰 발급 실패 {resp.status_code}: {resp.text[:200]}")
        data = resp.json()

    if "access_token" not in data:
        raise KISAuthError(f"응답에 access_token 없음: {data}")

    return TokenCache(
        access_token=data["access_token"],
        token_type=data.get("token_type", "Bearer"),
        issued_at=time.time(),
        expires_in=int(data.get("expires_in", 86400)),
    )


def get_kis_token(*, force_refresh: bool = False) -> TokenCache:
    """현재 KIS 모드(paper/live)에 맞는 유효 토큰 반환.

    캐시 우선, 만료 임박 또는 force_refresh 시 새로 발급.
    KIS는 일일 발급 한도가 있어 캐시는 강력 권장.
    """
    s = get_settings()
    cache_path = s.kis_token_cache_path

    if not force_refresh:
        cached = _load_cache(cache_path)
        if cached and cached.is_valid:
            logger.debug(
                f"KIS 토큰 캐시 사용 (모드={s.kis_mode.value}, "
                f"만료까지 {cached.remaining_seconds / 3600:.1f}h)"
            )
            return cached

    # 새 토큰 발급
    app_key = s.kis_app_key.get_secret_value()
    app_secret = s.kis_app_secret.get_secret_value()
    if not app_key or not app_secret:
        raise KISAuthError(
            f"KIS 자격증명 없음 (mode={s.kis_mode.value}). "
            ".env에 KIS_APP_KEY/KIS_APP_SECRET 설정 필요"
        )

    base_url = s.kis_base_url
    logger.info(f"KIS 토큰 신규 발급 시작 (모드={s.kis_mode.value}, base={base_url})")
    cache = _request_new_token(base_url=base_url, app_key=app_key, app_secret=app_secret)
    _save_cache(cache_path, cache)
    expiry = datetime.fromtimestamp(cache.expires_at).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"KIS 토큰 발급 성공 — 만료: {expiry}")
    return cache


def auth_headers(token: TokenCache | None = None) -> dict[str, str]:
    """KIS REST API 호출 시 사용할 인증 헤더."""
    if token is None:
        token = get_kis_token()
    s = get_settings()
    return {
        "Content-Type": "application/json",
        "authorization": f"{token.token_type} {token.access_token}",
        "appkey": s.kis_app_key.get_secret_value(),
        "appsecret": s.kis_app_secret.get_secret_value(),
    }


def is_paper_mode() -> bool:
    return get_settings().kis_mode is KISMode.PAPER
