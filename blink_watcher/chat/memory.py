"""What the chat agent keeps in mind, as markdown files in the chat area
(see chat.store), both safe to hand-edit:

    memory/memory.md       long-term memory (Memory): durable facts, one
                           "- (<date>) <fact>" bullet each, under a header
    <YYYY-MM-DD>/memory.md day notes (DayNotes): facts and requests that only
                           hold for that day, one "[HH:MM] <text>" line each

The agent adds entries and removes them by substring (remember / forget,
note_today / clear_today). Day notes roll over implicitly: each day has its
own file, so yesterday's entries stop being shown.

One-off requests in the day notes ("alert-on: <what>") are closed by the
notifier once an alert fulfils them (DayNotes.complete): the line becomes
"alert-done: <what> (<how>)" and no longer triggers alerts. Requests that hold
for the whole day ("alert-each: <what>") stay until cleared.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from threading import Lock

from blink_watcher.chat.store import DAY_NOTES, MEMORY_FILE, local_now

log = logging.getLogger(__name__)

MEMORY_HEADER = "# Chat agent long-term memory\n\n"
DAY_NOTES_SHOWN = 30  # the most recent lines of a day's notes that read() returns
REQUEST_LINE = re.compile(r"^(\[[^\]]*\]\s*)alert-on:\s*(.+?)\s*$", re.IGNORECASE)  # a one-off request


class Memory:
    """Long-term memory: chat/memory/memory.md."""

    def __init__(self, chat_dir: Path):
        self.path = chat_dir / MEMORY_FILE
        self._lock = Lock()

    def _raw(self) -> str:
        return self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    def read(self) -> str:
        """The entries, without the header."""
        raw = self._raw()
        return raw.removeprefix(MEMORY_HEADER).rstrip()

    def remember(self, note: str) -> str:
        note = note.strip()
        if not note:
            return "empty note; nothing saved"
        line = f"- ({datetime.now():%Y-%m-%d}) {note}\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._raw() or MEMORY_HEADER
            if not existing.endswith("\n"):
                existing += "\n"
            self.path.write_text(existing + line, encoding="utf-8")
        log.info("memory: remembered %r", note)
        return f"saved: {note}"

    def forget(self, substring: str) -> str:
        """Remove the entries ("- " lines) containing `substring` (any case)."""
        substring = substring.strip()
        if not substring:
            return "empty query; nothing removed"
        with self._lock:
            raw = self._raw()
            if not raw:
                return "memory is empty"
            kept, removed = _split(raw.splitlines(), substring, entries_only=True)
            if not removed:
                return f"no memory entries matched {substring!r}"
            self.path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        log.info("memory: forgot %d entry(ies) matching %r", len(removed), substring)
        return f"removed {len(removed)} entry(ies): " + "; ".join(r.lstrip("- ").strip() for r in removed)


class DayNotes:
    """Day notes: chat/<YYYY-MM-DD>/memory.md. `date` defaults to today in
    the configured timezone."""

    def __init__(self, chat_dir: Path, tz_name: str):
        self.chat_dir = chat_dir
        self.tz_name = tz_name
        self._lock = Lock()

    def _today(self) -> str:
        return local_now(self.tz_name).strftime("%Y-%m-%d")

    def _resolve(self, date: str | None) -> str:
        if not date:
            return self._today()
        date = date.strip()
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"invalid date {date!r}, expected YYYY-MM-DD") from e
        return date

    def path(self, date: str) -> Path:
        return self.chat_dir / DAY_NOTES.format(date=date)

    def read(self, date: str | None = None) -> str:
        """The day's most recent DAY_NOTES_SHOWN lines ("" if none)."""
        p = self.path(self._resolve(date))
        text = p.read_text(encoding="utf-8").rstrip() if p.exists() else ""
        return "\n".join(text.splitlines()[-DAY_NOTES_SHOWN:]) if text else ""

    def note(self, text: str, date: str | None = None, time: str | None = None) -> str:
        """Append "[HH:MM] text": `time` if given (the event's time), the
        current time for today, else "--:--"."""
        text = text.strip()
        if not text:
            return "empty note; nothing saved"
        resolved = self._resolve(date)
        if time:
            stamp = time.strip()
            try:
                datetime.strptime(stamp, "%H:%M")
            except ValueError as e:
                raise ValueError(f"invalid time {time!r}, expected HH:MM") from e
        elif resolved == self._today():
            stamp = local_now(self.tz_name).strftime("%H:%M")
        else:
            stamp = "--:--"
        with self._lock:
            p = self.path(resolved)
            p.parent.mkdir(parents=True, exist_ok=True)
            existing = p.read_text(encoding="utf-8") if p.exists() else ""
            if existing and not existing.endswith("\n"):
                existing += "\n"
            p.write_text(existing + f"[{stamp}] {text}\n", encoding="utf-8")
        log.info("daily: noted %r on %s", text, resolved)
        return f"noted on {resolved}: {text}"

    def clear(self, substring: str, date: str | None = None) -> str:
        """Remove the lines containing `substring` (any case); the file goes when empty."""
        substring = substring.strip()
        if not substring:
            return "empty query; nothing removed"
        resolved = self._resolve(date)
        with self._lock:
            p = self.path(resolved)
            if not p.exists():
                return f"{resolved} context is empty"
            kept, removed = _split(p.read_text(encoding="utf-8").splitlines(), substring)
            if not removed:
                return f"no entries matched {substring!r}"
            if kept:
                p.write_text("\n".join(kept) + "\n", encoding="utf-8")
            else:
                p.unlink()
        log.info("daily: cleared %d entry(ies) matching %r on %s", len(removed), substring, resolved)
        return f"removed {len(removed)} entry(ies) from {resolved}"


    def complete(self, requests: list[str], note: str, date: str | None = None) -> list[str]:
        """One-off requests fulfilled: each "alert-on:" line matching one of
        `requests` (either text containing the other, ignoring case and
        spacing) becomes "alert-done: <request> (<note>)". Returns the
        completed requests."""
        wanted = [_norm(r.split(":", 1)[1] if _norm(r).startswith("alert-on:") else r) for r in requests]
        wanted = [w for w in wanted if w]
        if not wanted:
            return []
        resolved = self._resolve(date)
        done = []
        with self._lock:
            p = self.path(resolved)
            if not p.exists():
                return []
            lines = p.read_text(encoding="utf-8").splitlines()
            for i, ln in enumerate(lines):
                m = REQUEST_LINE.match(ln)
                if m and any(w in _norm(m.group(2)) or _norm(m.group(2)) in w for w in wanted):
                    lines[i] = f"{m.group(1)}alert-done: {m.group(2)} ({note})"
                    done.append(m.group(2))
            if done:
                p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for r in done:
            log.info("daily: request completed on %s: %r (%s)", resolved, r, note)
        return done


def _norm(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .")


def _split(lines: list[str], substring: str, entries_only: bool = False) -> tuple[list[str], list[str]]:
    """(kept, removed): lines containing `substring` (any case) are removed;
    with entries_only only "- " bullet lines can be."""
    kept, removed = [], []
    for ln in lines:
        hit = substring.lower() in ln.lower() and (not entries_only or ln.startswith("- "))
        (removed if hit else kept).append(ln)
    return kept, removed
