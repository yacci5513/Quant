"""공용 로거 (loguru).

KST 타임스탬프 + 콘솔/파일 핸들러. 첫 import 시 1회 설정.

사용:
    from quant.common.logger import logger
    logger.info("hello")
"""

from __future__ import annotations

import sys

from loguru import logger as _logger

from quant.common.config import get_settings

_CONFIGURED = False


def _setup() -> None:
    """로거 핸들러 초기화 (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    _logger.remove()  # 기본 핸들러 제거

    # 콘솔: 색상, 사람이 읽기 좋게
    _logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
        backtrace=True,
        diagnose=settings.app_env.value == "development",
        enqueue=True,
    )

    # 파일: JSON-ish, 일자별 회전, 14일 보관
    _logger.add(
        settings.log_dir / "quant_{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        rotation="00:00",  # 자정마다 회전
        retention="14 days",
        compression="gz",
        encoding="utf-8",
        enqueue=True,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | " "{name}:{function}:{line} - {message}"
        ),
    )

    _CONFIGURED = True


_setup()

# 외부 사용자는 이 logger 인스턴스를 import
logger = _logger

__all__ = ["logger"]
