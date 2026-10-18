"""Application settings, from the environment and the project's .env.

Settings is composed of one section per area — storage, cameras, processing,
ai, chat, webui, stitch, security — plus each camera provider's own section, so an
area's options live next to the code that owns them. Field names are the
environment keys in lower case; .env.example documents every one of them
(the configuration page is built from it, see settings.document).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import SettingsConfigDict

from blink_watcher.core.paths import ENV_FILE
from blink_watcher.providers.blink.settings import BlinkSettings
from blink_watcher.settings.ai import AISettings
from blink_watcher.settings.cameras import ALL_CAMERAS, CameraSettings
from blink_watcher.settings.chat import ChatSettings
from blink_watcher.settings.security import SecuritySettings
from blink_watcher.settings.processing import ProcessingSettings
from blink_watcher.settings.stitch import StitchSettings
from blink_watcher.settings.storage import StorageSettings
from blink_watcher.settings.webui import WebUISettings

__all__ = ["ALL_CAMERAS", "Settings", "get_settings", "load_settings"]


class Settings(StorageSettings, CameraSettings, ProcessingSettings, AISettings, ChatSettings,
               WebUISettings, StitchSettings, SecuritySettings, BlinkSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    @property
    def chat_dir(self) -> Path:
        """Local folder of the chat area (memory, day notes, events index, chat history)."""
        from blink_watcher.core.storage import CHAT, make_storage
        return make_storage(self, CHAT).local_path("")


def load_settings() -> Settings:
    settings = Settings()
    if not settings.cameras:
        raise SystemExit('CAMERAS must list camera names, or be "all" for every camera.')
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """load_settings(), cached — for library code that needs a setting without
    threading Settings through every call."""
    return load_settings()
