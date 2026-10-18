"""The local web console."""
from __future__ import annotations

from pydantic_settings import BaseSettings


class WebUISettings(BaseSettings):
    webui_host: str = "127.0.0.1"
    webui_port: int = 9090
    webui_poll_sec: float = 3.0                 # how often the live view checks for new events
    webui_chat_lookback_days: int = 14          # days the console chat covers by default
    # Live monitor: hours snapshots are kept, and what the analyst sees of them.
    webui_live_keep_hours: float = 6.0
    webui_live_ai_frames: int = 3
    webui_live_ai_window_sec: float = 120.0
