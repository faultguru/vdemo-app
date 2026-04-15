import logging
from pathlib import Path
from typing import Optional

import aiohttp

from .config import Settings

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"


class TelegramError(RuntimeError):
    pass


class TelegramClient:
    def __init__(self, settings: Settings):
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            raise TelegramError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env"
            )
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id

    async def send_message(
        self,
        text: str,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
    ) -> dict:
        url = f"{API_BASE}/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_notification": disable_notification,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=30) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    raise TelegramError(f"sendMessage failed: {data}")
                log.debug("telegram sendMessage ok: %s", data.get("result", {}).get("message_id"))
                return data["result"]

    async def send_photo(self, photo: Path, caption: Optional[str] = None) -> dict:
        url = f"{API_BASE}/bot{self.token}/sendPhoto"
        form = aiohttp.FormData()
        form.add_field("chat_id", self.chat_id)
        if caption:
            form.add_field("caption", caption)
        form.add_field("photo", photo.open("rb"), filename=photo.name)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, timeout=120) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    raise TelegramError(f"sendPhoto failed: {data}")
                return data["result"]

    async def send_chat_action(self, action: str = "typing") -> None:
        url = f"{API_BASE}/bot{self.token}/sendChatAction"
        payload = {"chat_id": self.chat_id, "action": action}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    data = await resp.json()
                    if not data.get("ok"):
                        log.warning("sendChatAction not ok: %s", data)
                    else:
                        log.debug("sendChatAction ok: %s", action)
        except Exception as e:  # noqa: BLE001
            log.warning("sendChatAction error: %s", e)

    async def get_updates(
        self,
        offset: Optional[int] = None,
        timeout: int = 30,
    ) -> list[dict]:
        url = f"{API_BASE}/bot{self.token}/getUpdates"
        payload: dict = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=timeout + 10) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    raise TelegramError(f"getUpdates failed: {data}")
                return data.get("result", []) or []

    async def send_video(self, video: Path, caption: Optional[str] = None) -> dict:
        url = f"{API_BASE}/bot{self.token}/sendVideo"
        form = aiohttp.FormData()
        form.add_field("chat_id", self.chat_id)
        if caption:
            form.add_field("caption", caption)
        form.add_field("video", video.open("rb"), filename=video.name)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, timeout=300) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    raise TelegramError(f"sendVideo failed: {data}")
                return data["result"]
