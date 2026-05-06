"""애플리케이션 설정 (pydantic-settings).

`.env` 파일과 환경변수에서 설정을 로드한다.
민감 정보는 `SecretStr`로 마스킹된다.

사용:
    from quant.common.config import get_settings
    settings = get_settings()
    print(settings.kis_mode)
    print(settings.kis_base_url)             # 모드에 맞는 base URL 자동 선택
    print(settings.kis_app_key.get_secret_value())  # 마스킹 해제
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, enum.Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class KISMode(str, enum.Enum):
    """KIS API 운영 모드."""

    PAPER = "paper"  # 모의투자
    LIVE = "live"  # 실전투자


class Settings(BaseSettings):
    """전역 설정 컨테이너.

    환경변수 또는 `.env` 파일에서 로드. 컨테이너 내에선 docker-compose가
    `.env`를 주입하고, 로컬 실행 시엔 프로젝트 루트의 `.env`를 읽는다.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # ----- Runtime -----
    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = "INFO"
    tz: str = "Asia/Seoul"
    data_dir: Path = Path("/app/data")
    log_dir: Path = Path("/app/logs")

    # ----- KIS Developers -----
    kis_app_key: SecretStr = SecretStr("")
    kis_app_secret: SecretStr = SecretStr("")
    kis_account_no: str = ""
    kis_mode: KISMode = KISMode.PAPER

    kis_base_url_paper: str = "https://openapivts.koreainvestment.com:29443"
    kis_base_url_live: str = "https://openapi.koreainvestment.com:9443"

    # ----- DART (전자공시) -----
    dart_api_key: SecretStr = SecretStr("")

    # ---- Computed ----
    @property
    def kis_base_url(self) -> str:
        """현재 모드에 맞는 KIS base URL."""
        return self.kis_base_url_live if self.kis_mode is KISMode.LIVE else self.kis_base_url_paper

    @property
    def kis_token_cache_path(self) -> Path:
        """KIS 액세스 토큰 캐시 파일 경로.

        24h 유효 + 일일 발급 한도 → Volume에 영속화. 모드별 분리.
        """
        return self.data_dir / f".kis_token_{self.kis_mode.value}.json"

    # ---- Validation ----
    @model_validator(mode="after")
    def _check_live_credentials(self) -> Settings:
        """라이브 모드에선 KIS 자격증명이 반드시 있어야 한다.

        실수로 빈 자격증명으로 실거래 코드가 돌면 안 됨 → 기동 실패시킴.
        """
        if self.kis_mode is KISMode.LIVE:
            if not self.kis_app_key.get_secret_value():
                raise ValueError("KIS_MODE=live 인데 KIS_APP_KEY 가 비어있음")
            if not self.kis_app_secret.get_secret_value():
                raise ValueError("KIS_MODE=live 인데 KIS_APP_SECRET 가 비어있음")
            if not self.kis_account_no:
                raise ValueError("KIS_MODE=live 인데 KIS_ACCOUNT_NO 가 비어있음")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """싱글턴 Settings 헬퍼.

    `lru_cache`로 인스턴스를 메모이즈한다. 테스트에선
    `get_settings.cache_clear()`로 초기화 가능.
    """
    return Settings()  # type: ignore[call-arg]


__all__ = [
    "AppEnv",
    "KISMode",
    "Settings",
    "get_settings",
]
