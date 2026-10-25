"""Telegram Bot API client for the one authorized chat (TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID): messages, photos, videos, the typing indicator, and
long-polling for incoming messages."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import aiohttp

from blink_watcher.settings import Settings
from blink_watcher.settings import get_settings

_settings = get_settings()

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
TEXT_MAX, CAPTION_MAX = 4096, 1024   # Telegram rejects longer texts / captions
PHOTO_TIMEOUT_SEC = _settings.telegram_photo_timeout_sec
VIDEO_TIMEOUT_SEC = _settings.telegram_video_timeout_sec


class TelegramError(RuntimeError):
    """A Bot API error: `code` its error code (or the HTTP status), `retry_after`
    the seconds Telegram asked to wait (429)."""

    def __init__(self, message: str, code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after

    @property
    def permanent(self) -> bool:
        """Sending the same thing again fails again (a bad request, the bot blocked, a bad token)."""
        return self.code in (400, 401, 403, 404)


async def _result(method: str, resp: aiohttp.ClientResponse) -> dict:
    """The "result" of a Bot API response (TelegramError if not ok, e.g. a 502 page from a proxy)."""
    try:
        data = await resp.json(content_type=None)
    except ValueError:
        raise TelegramError(f"{method} failed: HTTP {resp.status}", code=resp.status) from None
    if not data.get("ok"):
        raise TelegramError(f"{method} failed: {data.get('error_code')} {data.get('description')}",
                            code=data.get("error_code"), retry_after=(data.get("parameters") or {}).get("retry_after"))
    return data["result"]


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


class TelegramClient:
    def __init__(self, settings: Settings):
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            raise TelegramError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id

    async def _call(self, method: str, payload: dict, timeout: int) -> dict:
        """A JSON Bot API call; its "result" (TelegramError if not ok)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{API_BASE}/bot{self.token}/{method}", json=payload, timeout=timeout) as resp:
                return await _result(method, resp)

    async def send_message(self, text: str) -> dict:
        return await self._call("sendMessage", {"chat_id": self.chat_id, "text": _cut(text, TEXT_MAX)}, timeout=30)

    async def send_photo(self, photo: Path, caption: Optional[str] = None) -> dict:
        return await self._send_file("sendPhoto", "photo", photo, caption, timeout=PHOTO_TIMEOUT_SEC)

    async def send_video(self, video: Path, caption: Optional[str] = None) -> dict:
        return await self._send_file("sendVideo", "video", video, caption, timeout=VIDEO_TIMEOUT_SEC)

    async def _send_file(self, method: str, field: str, path: Path, caption: Optional[str], timeout: int) -> dict:
        form = aiohttp.FormData()
        form.add_field("chat_id", self.chat_id)
        if caption:
            form.add_field("caption", _cut(caption, CAPTION_MAX))
        with path.open("rb") as f:
            form.add_field(field, f, filename=path.name)
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{API_BASE}/bot{self.token}/{method}", data=form, timeout=timeout) as resp:
                    return await _result(method, resp)

    async def send_chat_action(self, action: str = "typing") -> None:
        """Best effort: a failure is only logged (debug: the reply that follows reports a real problem)."""
        try:
            await self._call("sendChatAction", {"chat_id": self.chat_id, "action": action}, timeout=10)
        except Exception as e:  # noqa: BLE001
            log.debug("sendChatAction error: %s", e)

    async def get_updates(self, offset: Optional[int] = None, timeout: int = 30) -> list[dict]:
        """Incoming messages after `offset` (long poll, up to `timeout` seconds)."""
        payload: dict = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        return await self._call("getUpdates", payload, timeout=timeout + 10) or []
