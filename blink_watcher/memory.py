"""Long-term memory for the chat agent, persisted as a markdown file.

The file is append-only from the agent's perspective: the agent can add
bulleted entries and remove entries by substring. It lives at
``events/memory.md`` and is safe to hand-edit by the user.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from threading import Lock

log = logging.getLogger(__name__)

MEMORY_HEADER = "# Chat agent long-term memory\n\n"


class Memory:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()

    def read(self) -> str:
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")

    def read_body(self) -> str:
        """Memory content without the header."""
        raw = self.read()
        if raw.startswith(MEMORY_HEADER):
            return raw[len(MEMORY_HEADER):].rstrip()
        return raw.rstrip()

    def remember(self, note: str) -> str:
        note = note.strip()
        if not note:
            return "empty note; nothing saved"
        ts = datetime.now().strftime("%Y-%m-%d")
        line = f"- ({ts}) {note}\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existing = self.read()
            if not existing:
                existing = MEMORY_HEADER
            if not existing.endswith("\n"):
                existing += "\n"
            self.path.write_text(existing + line, encoding="utf-8")
        log.info("memory: remembered %r", note)
        return f"saved: {note}"

    def forget(self, substring: str) -> str:
        substring = substring.strip()
        if not substring:
            return "empty query; nothing removed"
        with self._lock:
            raw = self.read()
            if not raw:
                return "memory is empty"
            lines = raw.splitlines()
            kept: list[str] = []
            removed: list[str] = []
            for ln in lines:
                if ln.startswith("- ") and substring.lower() in ln.lower():
                    removed.append(ln)
                else:
                    kept.append(ln)
            if not removed:
                return f"no memory entries matched {substring!r}"
            self.path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        log.info("memory: forgot %d entry(ies) matching %r", len(removed), substring)
        return f"removed {len(removed)} entry(ies): " + "; ".join(
            r.lstrip("- ").strip() for r in removed
        )
