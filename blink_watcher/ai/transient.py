"""Recognising transient failures in a provider's error text."""
from __future__ import annotations

import re
from typing import Optional

_TRANSIENT = re.compile(
    r"\b(429|500|502|503|504|529)\b|rate.?limit|usage limit|limit reached|overloaded|too many requests"
    r"|timed? ?out|temporarily unavailable|connection (error|reset)|ECONNRESET",
    re.IGNORECASE,
)


def is_transient(text: Optional[str]) -> bool:
    return bool(text and _TRANSIENT.search(text))


def one_line(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."
