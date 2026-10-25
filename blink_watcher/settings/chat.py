"""The chat agent (Telegram chat and event notifications) and Telegram."""
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings

from blink_watcher.settings.ai import Effort


class ChatSettings(BaseSettings):
    event_process_limit: int = 1                # events per notifier run; 0 = no limit
    # Events recorded longer ago than this (a catch-up, a backlog) are recorded but not
    # alerted, so Telegram never gets old news as if it were happening now; 0 = no limit.
    notify_max_age_minutes: int = 60
    # Every round goes to the fast model; a round that changes memory or day notes
    # is redone by the memory model. Notifications never change memory.
    chat_agent_model: str = "claude-sonnet-5-5"
    chat_agent_effort: Effort = "low"
    chat_agent_memory_model: str = "claude-opus-5-5"
    chat_agent_memory_effort: Effort = "medium"

    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    # Tuning (.env.example documents each option):
    chat_max_tool_rounds: int = 6
    chat_max_pictures: int = 6
    chat_lookback_days: int = 14
    chat_history_window: int = 10
    chat_media_offer_cooldown_sec: int = 3600
    day_notes_shown: int = 30
    day_notes_upcoming_days: int = 31
    telegram_outbox_poll_sec: float = 2.0
    telegram_retry_sec: float = 5.0
    telegram_retry_max_sec: float = 300.0
    telegram_photo_timeout_sec: int = 120
    telegram_video_timeout_sec: int = 300
