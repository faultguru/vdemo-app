"""The LLM provider behind the video describer and the chat agent, and its retry policy."""
from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings

LLMProvider = Literal["claude-cli", "claude-api"]
Effort = Literal["", "low", "medium", "high", "xhigh", "max"]


class AISettings(BaseSettings):
    llm_provider: LLMProvider = "claude-cli"
    # A throttled / overloaded / timed-out request waits this long and is retried,
    # up to LLM_MAX_RETRIES times (0 = until it goes through).
    llm_retry_sleep_sec: int = 30
    llm_max_retries: int = 20
    # The video describer: sampled key frames of a clip -> structured description.
    describer_model: str = "claude-sonnet-5-5"
    describer_effort: Effort = ""  # "" = the model's default
    # Tuning (.env.example documents each option):
    claude_cli_timeout_sec: int = 900
    claude_api_max_tokens: int = 16000
    llm_image_max_side: int = 1024
    gemini_trim_pad_sec: float = 0.5
    gemini_min_trim_saving_sec: float = 1.0
    gemini_upload_timeout_sec: int = 120
    gemini_person_fps: float = 5.0
    gemini_default_fps: float = 1.0
