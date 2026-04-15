"""Ephemeral per-day scratchpad shared by the chat agent and the
notification composer.

Entries live in ``state_dir/daily/YYYY-MM-DD.md`` and look like::

    [09:12] waiting for pizza delivery
    [14:12] package dropped on front porch — brown box, FedEx van
    [14:35] adult male entered side door — matches household

Rollover is implicit: the file is always the one named after "today" in
the configured timezone, so yesterday's entries simply stop being read.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)


class DailyContext:
    def __init__(self, base_dir: Path, tz: ZoneInfo | None):
        self.base_dir = base_dir
        self.tz = tz
        self._lock = Lock()

    def _now(self) -> datetime:
        return datetime.now(self.tz) if self.tz else datetime.now().astimezone()

    def _path_for_date(self, date: str) -> Path:
        return self.base_dir / f"{date}.md"

    def _today_str(self) -> str:
        return self._now().strftime("%Y-%m-%d")

    def _resolve_date(self, date: str | None) -> str:
        if not date:
            return self._today_str()
        date = date.strip()
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"invalid date {date!r}, expected YYYY-MM-DD") from e
        return date

    def read_body(self, max_lines: int = 30, date: str | None = None) -> str:
        """Return a day's entries, capped at the most recent ``max_lines``.

        Defaults to today when ``date`` is None.
        """
        resolved = self._resolve_date(date)
        p = self._path_for_date(resolved)
        if not p.exists():
            return ""
        text = p.read_text(encoding="utf-8").rstrip()
        if not text:
            return ""
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return "\n".join(lines)

    def note(
        self,
        text: str,
        date: str | None = None,
        time: str | None = None,
    ) -> str:
        text = text.strip()
        if not text:
            return "empty note; nothing saved"
        resolved = self._resolve_date(date)
        if time:
            stamp = time.strip()
            try:
                datetime.strptime(stamp, "%H:%M")
            except ValueError as e:
                raise ValueError(f"invalid time {time!r}, expected HH:MM") from e
        elif resolved == self._today_str():
            stamp = self._now().strftime("%H:%M")
        else:
            stamp = "--:--"
        line = f"[{stamp}] {text}\n"
        with self._lock:
            p = self._path_for_date(resolved)
            p.parent.mkdir(parents=True, exist_ok=True)
            existing = p.read_text(encoding="utf-8") if p.exists() else ""
            if existing and not existing.endswith("\n"):
                existing += "\n"
            p.write_text(existing + line, encoding="utf-8")
        log.info("daily: noted %r on %s", text, resolved)
        return f"noted on {resolved}: {text}"

    def clear(self, substring: str, date: str | None = None) -> str:
        substring = substring.strip()
        if not substring:
            return "empty query; nothing removed"
        resolved = self._resolve_date(date)
        with self._lock:
            p = self._path_for_date(resolved)
            if not p.exists():
                return f"{resolved} context is empty"
            lines = p.read_text(encoding="utf-8").splitlines()
            kept: list[str] = []
            removed: list[str] = []
            for ln in lines:
                if substring.lower() in ln.lower():
                    removed.append(ln)
                else:
                    kept.append(ln)
            if not removed:
                return f"no entries matched {substring!r}"
            if kept:
                p.write_text("\n".join(kept) + "\n", encoding="utf-8")
            else:
                p.unlink()
        log.info("daily: cleared %d entry(ies) matching %r on %s",
                 len(removed), substring, resolved)
        return f"removed {len(removed)} entry(ies) from {resolved}"
