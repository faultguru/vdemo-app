"""Claude through the Messages API (Anthropic SDK), authenticated with the
OAuth token of the Claude Code login (~/.claude/.credentials.json).

The request body is written to the request folder as llm.request.api.json
(structured output as output_config.format, the wire form of messages.parse).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from blink_watcher.ai.base import LLMClient, LLMRequest, LLMResponse, LLMThrottled, RetryPolicy, Usage
from blink_watcher.ai.claude_cli.client import content_blocks
from blink_watcher.ai.transient import one_line

REQUEST_API = "llm.request.api.json"
MAX_TOKENS = 16000  # thinking is on by default on current models; room for it plus the answer
CREDENTIALS = Path.home() / ".claude" / ".credentials.json"
TRANSIENT_STATUS = (429, 500, 502, 503, 504, 529)


def _oauth_token() -> str:
    if not CREDENTIALS.exists():
        raise SystemExit(f"Claude credentials not found at {CREDENTIALS}. Run `claude` once to sign in.")
    token = (json.loads(CREDENTIALS.read_text(encoding="utf-8")).get("claudeAiOauth") or {}).get("accessToken")
    if not token:
        raise SystemExit(f"No accessToken in {CREDENTIALS}")
    return token


def _system(request: LLMRequest) -> list[dict]:
    return [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]


def _messages(request: LLMRequest) -> list[dict]:
    return [{"role": "user", "content": content_blocks(request)}]


class ClaudeApiClient(LLMClient):
    name = "claude-api"

    def __init__(self, retry: RetryPolicy = RetryPolicy()):
        super().__init__(retry)
        import anthropic

        self._anthropic = anthropic
        os.environ.pop("ANTHROPIC_API_KEY", None)  # the OAuth token must be the credential used
        self._client = anthropic.Anthropic(auth_token=_oauth_token(),
                                           default_headers={"anthropic-beta": "oauth-2025-04-20"})

    def write_request(self, request: LLMRequest, request_dir: Path) -> list[str]:
        body = {"model": request.model, "max_tokens": MAX_TOKENS, "system": _system(request),
                "messages": _messages(request),
                "output_config": {"format": {"type": "json_schema", "schema": request.schema.model_json_schema()}}}
        (request_dir / REQUEST_API).write_text(json.dumps(body, indent=2), encoding="utf-8")
        return [REQUEST_API]

    def _send(self, request: LLMRequest, request_dir: Optional[Path]) -> LLMResponse:
        anthropic = self._anthropic
        system, messages = _system(request), _messages(request)
        if request_dir:
            self.write_request(request, request_dir)
        try:
            result = self._client.messages.parse(model=request.model, max_tokens=MAX_TOKENS, system=system,
                                                 messages=messages, output_format=request.schema)
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise LLMThrottled(one_line(f"{type(e).__name__}: {e}")) from e
        except anthropic.APIStatusError as e:
            if e.status_code in TRANSIENT_STATUS:
                raise LLMThrottled(one_line(f"HTTP {e.status_code}: {e}")) from e
            raise
        parsed = result.parsed_output
        return LLMResponse(
            parsed=parsed, provider=self.name, model=request.model, effort=request.effort,
            stop_reason=result.stop_reason, usage=Usage.from_dict(result.usage.model_dump()),
            request_id=getattr(result, "_request_id", None),
            raw_text=None if parsed is not None else "".join(
                getattr(b, "text", "") for b in result.content if getattr(b, "type", "") == "text"),
            request_files=[REQUEST_API] if request_dir else [],
        )
