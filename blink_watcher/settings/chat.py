"""The chat agent (Telegram chat and event notifications) and Telegram."""
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings

from blink_watcher.settings.ai import Effort


class ChatSettings(BaseSettings):
    event_process_limit: int = 1                # events per notifier run; 0 = no limit
    # Every round goes to the fast model; a round that changes memory or day notes
    # is redone by the memory model. Notifications never change memory.
    chat_agent_model: str = "claude-sonnet-5-5"
    chat_agent_effort: Effort = "low"
    chat_agent_memory_model: str = "claude-opus-5-5"
    chat_agent_memory_effort: Effort = "medium"

    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
