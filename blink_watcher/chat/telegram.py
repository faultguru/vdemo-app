"""Telegram Bot API client for the one authorized chat (TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID): messages, photos, videos, the typing indicator, and
long-polling for incoming messages."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import aiohttp

from blink_watcher.settings import Settings

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"


class TelegramError(RuntimeError):
    pass


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
                data = await resp.json()
        if not data.get("ok"):
            raise TelegramError(f"{method} failed: {data}")
        return data["result"]

    async def send_message(self, text: str) -> dict:
        return await self._call("sendMessage", {"chat_id": self.chat_id, "text": text}, timeout=30)

    async def send_photo(self, photo: Path, caption: Optional[str] = None) -> dict:
        return await self._send_file("sendPhoto", "photo", photo, caption, timeout=120)

    async def send_video(self, video: Path, caption: Optional[str] = None) -> dict:
        return await self._send_file("sendVideo", "video", video, caption, timeout=300)

    async def _send_file(self, method: str, field: str, path: Path, caption: Optional[str], timeout: int) -> dict:
        form = aiohttp.FormData()
        form.add_field("chat_id", self.chat_id)
        if caption:
            form.add_field("caption", caption)
        with path.open("rb") as f:
            form.add_field(field, f, filename=path.name)
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{API_BASE}/bot{self.token}/{method}", data=form, timeout=timeout) as resp:
                    data = await resp.json()
        if not data.get("ok"):
            raise TelegramError(f"{method} failed: {data}")
        return data["result"]

    async def send_chat_action(self, action: str = "typing") -> None:
        """Best effort: a failure is only logged."""
        try:
            await self._call("sendChatAction", {"chat_id": self.chat_id, "action": action}, timeout=10)
        except Exception as e:  # noqa: BLE001
            log.warning("sendChatAction error: %s", e)

    async def get_updates(self, offset: Optional[int] = None, timeout: int = 30) -> list[dict]:
        """Incoming messages after `offset` (long poll, up to `timeout` seconds)."""
        payload: dict = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        return await self._call("getUpdates", payload, timeout=timeout + 10) or []
