"""LLM providers behind one interface (ai.base). make_llm() returns the one
LLM_PROVIDER names; the describer and the chat agent never import a
provider directly.

  claude_cli   Claude via `claude -p` (Claude Code login)       LLM_PROVIDER=claude-cli
  claude_api   Claude via the Messages API (same login's token)  LLM_PROVIDER=claude-api
  gemini       reference video describer on Gemini (not wired to settings)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from blink_watcher.ai.base import (
    ImagePart, LLMClient, LLMRequest, LLMResponse, LLMThrottled, Part, RetryPolicy, TextPart, Usage,
)

if TYPE_CHECKING:
    from blink_watcher.settings import Settings

__all__ = ["ImagePart", "LLMClient", "LLMRequest", "LLMResponse", "LLMThrottled", "Part", "RetryPolicy",
           "TextPart", "Usage", "make_llm"]


def make_llm(settings: "Settings") -> LLMClient:
    """The provider, reporting each request to the ai dependency (core.health)."""
    from blink_watcher.core.health import Dependency
    retry = RetryPolicy(settings.llm_retry_sleep_sec, settings.llm_max_retries)
    if settings.llm_provider == "claude-cli":
        from blink_watcher.ai.claude_cli import ClaudeCliClient
        client: LLMClient = ClaudeCliClient(retry)
    elif settings.llm_provider == "claude-api":
        from blink_watcher.ai.claude_api import ClaudeApiClient
        client = ClaudeApiClient(retry)
    else:
        raise ValueError(f"unknown LLM_PROVIDER {settings.llm_provider!r}")
    client.health = Dependency(settings, "ai", label=f"AI ({settings.llm_provider})")
    return client
