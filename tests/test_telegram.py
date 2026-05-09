"""텔레그램 알림 모듈 테스트 (네트워크 미사용)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from quant.notify.telegram import _split_long, send_telegram


def test_split_long_short_passthrough() -> None:
    text = "짧은 메시지"
    assert _split_long(text) == [text]


def test_split_long_breaks_on_lines() -> None:
    lines = [f"line {i}" * 100 for i in range(50)]
    text = "\n".join(lines)
    chunks = _split_long(text, max_len=2000)
    assert len(chunks) > 1
    # 각 청크가 한도 이하
    assert all(len(c) <= 2000 for c in chunks)
    # 합치면 원본 (개행 보존)
    rejoined = "\n".join(chunks)
    assert rejoined == text


def test_send_telegram_skips_when_unconfigured() -> None:
    """토큰/chat_id 비면 silent skip."""
    result = send_telegram("test", token="", chat_id="", silent_if_unconfigured=True)
    assert result is False


def test_send_telegram_raises_when_unconfigured_strict() -> None:
    """silent_if_unconfigured=False면 예외."""
    with pytest.raises(ValueError, match="TELEGRAM"):
        send_telegram("test", token="", chat_id="", silent_if_unconfigured=False)


def test_send_telegram_calls_api(monkeypatch) -> None:
    """모킹된 _send_one이 호출되는지."""
    calls = []

    def fake_send(token, chat_id, text):
        calls.append((token, chat_id, text))

    with patch("quant.notify.telegram._send_one", side_effect=fake_send):
        ok = send_telegram(
            "hello", token="DUMMY_TOKEN", chat_id="123456", silent_if_unconfigured=False
        )
    assert ok is True
    assert len(calls) == 1
    assert calls[0][2] == "hello"


def test_send_telegram_chunks_long(monkeypatch) -> None:
    """4000자 초과 메시지는 여러 청크로 나뉨."""
    calls = []

    def fake_send(token, chat_id, text):
        calls.append(text)

    long_text = "\n".join([f"line{i:04d}" for i in range(800)])  # ~6400자
    with patch("quant.notify.telegram._send_one", side_effect=fake_send):
        send_telegram(long_text, token="T", chat_id="C", silent_if_unconfigured=False)
    assert len(calls) >= 2
