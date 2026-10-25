"""The LLM provider interface: one structured-output request — a system prompt,
text and image parts, a pydantic schema for the answer — and its normalised
response. The video describer and the chat agent are written against this
interface only; each provider (claude_cli, claude_api, ...) translates the
request into its own wire format and handles its own credentials.

Retrying is shared: a provider raises LLMThrottled for a transient failure
(rate or usage limit, overload, 5xx, timeout, dropped connection) and
LLMClient.ask() waits and tries again, per the RetryPolicy. Each request's
outcome is reported to `health` (the ai dependency, core.health) when set:
failed once it gives up, ok once it is answered.
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from pydantic import BaseModel

log = logging.getLogger(__name__)

# progress(message, logging level): one-line status notices (retries, ...) for the caller's log.
Progress = Callable[..., None]


# ---------- request ----------

@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str = "image/jpeg"


Part = Union[TextPart, ImagePart]


@dataclass
class LLMRequest:
    system: str
    parts: list[Part]
    schema: type[BaseModel]   # the answer's shape (structured output)
    model: str
    effort: Optional[str] = None  # low | medium | high | xhigh | max; None = the model's default


# ---------- response ----------

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    def add(self, other: "Usage") -> None:
        for name in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Usage":
        d = d or {}
        return cls(**{k: int(d.get(k) or 0) for k in cls.__dataclass_fields__})


@dataclass
class LLMResponse:
    parsed: Optional[BaseModel]       # None: the model gave no valid structured answer (see raw_text)
    provider: str
    model: str
    effort: Optional[str] = None
    stop_reason: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: Optional[float] = None  # what the request would cost at API prices, if the provider reports it
    request_id: Optional[str] = None
    raw_text: Optional[str] = None
    duration_ms: int = 0              # including retry waits
    retries: int = 0
    request_files: list[str] = field(default_factory=list)

    def record(self) -> dict:
        """The call as stored in metadata files (llm block, round records). Keys
        kept from the earlier format: backend (= provider), cost_usd_equivalent."""
        d = asdict(self)
        d.pop("parsed")
        d["backend"] = d["provider"]
        d["cost_usd_equivalent"] = d.pop("cost_usd")
        return d


class LLMThrottled(RuntimeError):
    """A transient failure; the request can be retried."""


@dataclass(frozen=True)
class RetryPolicy:
    sleep_sec: float = 30
    max_retries: int = 20  # 0 = retry until it goes through


# ---------- client ----------

class LLMClient(ABC):
    """A provider. Subclasses implement _send (one attempt)."""

    name: str
    health = None  # core.health.Dependency: set by make_llm()

    def __init__(self, retry: RetryPolicy = RetryPolicy()):
        self.retry = retry

    @abstractmethod
    def write_request(self, request: LLMRequest, request_dir: Path) -> list[str]:
        """Write the exact request to request_dir so it can be inspected or
        replayed, without sending it; returns the file names."""

    @abstractmethod
    def _send(self, request: LLMRequest, request_dir: Optional[Path]) -> LLMResponse:
        """One attempt. Writes the exact request to request_dir (if given, see
        write_request); raises LLMThrottled on a transient failure."""

    def ask(self, request: LLMRequest, request_dir: Optional[Path] = None,
            progress: Optional[Progress] = None) -> LLMResponse:
        """Send the request, retrying throttled attempts. Blocking."""
        try:
            response = self._ask(request, request_dir, progress)
        except Exception as e:
            if self.health:
                self.health.failed(f"{type(e).__name__}: {e}" if str(e) else type(e).__name__)
            raise
        if self.health:
            self.health.ok()
        return response

    def _ask(self, request: LLMRequest, request_dir: Optional[Path],
             progress: Optional[Progress]) -> LLMResponse:
        progress = progress or (lambda msg, level=logging.INFO: log.log(level, msg))
        if request_dir:
            request_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        retries = 0
        while True:
            try:
                response = self._send(request, request_dir)
                break
            except LLMThrottled as e:
                limit = self.retry.max_retries
                if limit and retries >= limit:
                    progress(f"{self.name}: request throttled ({e}) - retry limit reached ({limit}), giving up",
                             logging.ERROR)
                    raise
                retries += 1
                progress(f"{self.name}: request throttled ({e}) - retrying in {self.retry.sleep_sec:g}s "
                         f"(retry {retries}/{limit or 'unlimited'})", logging.WARNING)
                time.sleep(self.retry.sleep_sec)
        response.duration_ms = int((time.perf_counter() - t0) * 1000)
        response.retries = retries
        return response

    async def ask_async(self, request: LLMRequest, request_dir: Optional[Path] = None,
                        progress: Optional[Progress] = None) -> LLMResponse:
        return await asyncio.to_thread(self.ask, request, request_dir, progress)
