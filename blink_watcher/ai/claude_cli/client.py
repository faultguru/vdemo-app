"""Claude through the headless Claude Code CLI (`claude -p`), on the user's
Claude Code login.

The request runs with the given system prompt only — no tools, no MCP, no
project settings or memory (from an empty working folder) — and the answer
schema as --json-schema. The exact request is written to the request folder:

  llm.request.jsonl       stdin: one stream-json user message (text + images)
  llm.request.args.json   argv (minus the executable)
  llm.system.txt          the system prompt (a file: argv is capped at 32K chars on Windows)

Replay by hand: `claude <args...> < llm.request.jsonl`.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from blink_watcher.ai.base import ImagePart, LLMClient, LLMRequest, LLMResponse, LLMThrottled, TextPart, Usage
from blink_watcher.ai.transient import is_transient, one_line
from blink_watcher.settings import get_settings

_settings = get_settings()

REQUEST_JSONL = "llm.request.jsonl"
REQUEST_ARGS = "llm.request.args.json"
REQUEST_SYSTEM = "llm.system.txt"
TIMEOUT_SEC = _settings.claude_cli_timeout_sec


def content_blocks(request: LLMRequest) -> list[dict]:
    """The request's parts as Claude content blocks."""
    blocks = []
    for part in request.parts:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": part.media_type,
                                                       "data": base64.b64encode(part.data).decode()}})
    return blocks


class ClaudeCliClient(LLMClient):
    name = "claude-cli"

    def _args(self, request: LLMRequest, system_file: Path) -> list[str]:
        return [
            "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json", "--verbose",
            "--model", request.model,
            "--system-prompt-file", str(system_file.resolve()),
            "--json-schema", json.dumps(request.schema.model_json_schema()),
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--no-session-persistence",
            *(["--effort", request.effort] if request.effort else []),
        ]

    def write_request(self, request: LLMRequest, request_dir: Path) -> list[str]:
        (request_dir / REQUEST_JSONL).write_text(json.dumps({"type": "user", "message": {
            "role": "user", "content": content_blocks(request)}}) + "\n", encoding="utf-8")
        system_file = request_dir / REQUEST_SYSTEM
        system_file.write_text(request.system, encoding="utf-8")
        (request_dir / REQUEST_ARGS).write_text(json.dumps(self._args(request, system_file), indent=2),
                                                encoding="utf-8")
        return [REQUEST_JSONL, REQUEST_ARGS, REQUEST_SYSTEM]

    def _send(self, request: LLMRequest, request_dir: Optional[Path]) -> LLMResponse:
        exe = shutil.which("claude")
        if not exe:
            raise SystemExit("`claude` CLI not found on PATH")
        with tempfile.TemporaryDirectory(prefix="llm-", ignore_cleanup_errors=True) as tmp:
            folder = request_dir or Path(tmp)  # the CLI reads stdin and the system prompt from files
            files = self.write_request(request, folder)
            args = json.loads((folder / REQUEST_ARGS).read_text(encoding="utf-8"))
            with tempfile.TemporaryDirectory(prefix="llm-cwd-", ignore_cleanup_errors=True) as cwd, \
                    open(folder / REQUEST_JSONL, encoding="utf-8") as stdin:
                try:
                    proc = subprocess.run([exe, *args], stdin=stdin, capture_output=True, text=True,
                                          encoding="utf-8", cwd=cwd, timeout=TIMEOUT_SEC)
                except subprocess.TimeoutExpired as e:
                    raise LLMThrottled(f"claude -p timed out after {TIMEOUT_SEC}s") from e
        result = _result_event(proc.stdout)
        if result is None:
            if is_transient(proc.stderr) or is_transient(proc.stdout):
                raise LLMThrottled(one_line(f"exit {proc.returncode}: {proc.stderr or proc.stdout}"))
            raise RuntimeError(f"claude -p exited {proc.returncode} without a result event:\n"
                               f"stderr: {proc.stderr[-2000:]}\nstdout: {proc.stdout[-2000:]}")
        if result.get("is_error") and is_transient(result.get("result")):
            raise LLMThrottled(one_line(result.get("result")))
        parsed = _parse(request, result)
        return LLMResponse(
            parsed=parsed, provider=self.name, model=request.model, effort=request.effort,
            stop_reason=result.get("stop_reason") or result.get("subtype"),
            usage=Usage.from_dict(result.get("usage")), cost_usd=result.get("total_cost_usd"),
            request_id=result.get("session_id"),
            raw_text=None if parsed is not None else result.get("result"),
            request_files=files if request_dir else [],
        )


def _result_event(stdout: str) -> Optional[dict]:
    result = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
    return result


def _parse(request: LLMRequest, result: dict):
    structured = result.get("structured_output")
    if structured is not None:
        return request.schema.model_validate(structured)
    if result.get("result"):
        try:
            return request.schema.model_validate_json(result["result"])
        except ValueError:
            return None
    return None
