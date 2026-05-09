"""텔레그램 봇 알림.

사용:
    from quant.notify.telegram import send_telegram
    send_telegram("매일 알림 텍스트")

config:
- TELEGRAM_BOT_TOKEN  : @BotFather에서 발급
- TELEGRAM_CHAT_ID    : 봇과 대화 후 getUpdates에서 확인

토큰·chat_id 비어있으면 silently skip (개발 환경 안전).
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from quant.common.config import get_settings
from quant.common.logger import logger

_API_BASE = "https://api.telegram.org"
_DEFAULT_TIMEOUT = 10.0
_MAX_LEN = 4000  # Telegram 메시지 한도 4096, 여유 96자


class TelegramError(RuntimeError):
    """Telegram API 호출 실패."""


def _split_long(text: str, max_len: int = _MAX_LEN) -> list[str]:
    """긴 메시지를 라인 경계 기준으로 분할."""
    if len(text) <= max_len:
        return [text]
    lines = text.split("\n")
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in lines:
        if cur_len + len(line) + 1 > max_len and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [line], len(line)
        else:
            cur.append(line)
            cur_len += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _send_one(token: str, chat_id: str, text: str) -> None:
    url = f"{_API_BASE}/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            raise TelegramError(f"Telegram API {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if not data.get("ok"):
            raise TelegramError(f"Telegram error: {data.get('description', '')[:200]}")


def send_telegram(
    text: str,
    *,
    token: str | None = None,
    chat_id: str | None = None,
    silent_if_unconfigured: bool = True,
) -> bool:
    """메시지 전송. 토큰 미설정 시 silent skip (silent_if_unconfigured=True).

    Returns:
        True = 발송 성공, False = 미설정으로 skip.
    """
    if token is None or chat_id is None:
        s = get_settings()
        token = token or s.telegram_bot_token.get_secret_value()
        chat_id = chat_id or s.telegram_chat_id

    if not token or not chat_id:
        if silent_if_unconfigured:
            logger.debug("Telegram 미설정 — 메시지 skip")
            return False
        raise ValueError("TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID 비어있음")

    chunks = _split_long(text)
    for chunk in chunks:
        _send_one(token, chat_id, chunk)
    logger.info(f"Telegram 발송: {len(chunks)}건 chunk, {len(text)}자")
    return True


def send_test_ping(token: str | None = None, chat_id: str | None = None) -> bool:
    """연결 테스트용 ping. CLI에서 사용."""
    return send_telegram(
        "✅ Quant 봇 연결 테스트 — 알림이 정상 작동합니다",
        token=token,
        chat_id=chat_id,
        silent_if_unconfigured=False,
    )
