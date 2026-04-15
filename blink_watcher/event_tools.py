"""Tools exposed to the chat agent for querying/sending Blink events.

All file access is sandboxed: any path resolved from event JSON must live
under ``settings.events_dir``. Paths escaping that folder are rejected.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .config import Settings
from .telegram import TelegramClient

log = logging.getLogger(__name__)


class EventToolError(Exception):
    pass


class EventTools:
    def __init__(self, settings: Settings, telegram: TelegramClient):
        self.settings = settings
        self.telegram = telegram
        self._events_root = settings.events_dir.resolve()

    # ---------- sandbox helpers ----------

    def resolve_under_events(self, raw: str) -> Path:
        """Public sandbox resolver. Raises EventToolError if path escapes."""
        return self._sandbox(raw)

    def _sandbox(self, raw: str) -> Path:
        candidate = (Path.cwd() / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        try:
            candidate.relative_to(self._events_root)
        except ValueError as e:
            raise EventToolError(f"path escapes events sandbox: {raw}") from e
        if not candidate.exists():
            raise EventToolError(f"file not found: {raw}")
        return candidate

    def _events_file(self, date: str) -> Path:
        path = (self._events_root / f"events.{date}.json").resolve()
        try:
            path.relative_to(self._events_root)
        except ValueError as e:
            raise EventToolError(f"invalid date: {date}") from e
        if not path.exists():
            raise EventToolError(f"no events file for {date}")
        return path

    def _load_date(self, date: str) -> list[dict]:
        return json.loads(self._events_file(date).read_text(encoding="utf-8"))

    def _find_event(self, event_id: str, date: Optional[str] = None) -> dict:
        dates = [date] if date else self.list_event_dates()
        for d in dates:
            try:
                for ev in self._load_date(d):
                    if ev.get("event_id") == event_id:
                        return ev
            except EventToolError:
                continue
        raise EventToolError(f"event not found: {event_id}")

    # ---------- tool methods ----------

    def list_event_dates(self) -> list[str]:
        if not self._events_root.exists():
            return []
        out: list[str] = []
        for p in sorted(self._events_root.glob("events.*.json")):
            name = p.name
            date = name[len("events."):-len(".json")]
            out.append(date)
        return out

    def list_events(self, date: str) -> list[dict]:
        compact: list[dict] = []
        for ev in self._load_date(date):
            summary = ev.get("summary") or {}
            compact.append({
                "event_id": ev.get("event_id"),
                "recorded_at": ev.get("recorded_at"),
                "description": summary.get("description", ""),
                "people_types": summary.get("people_types", []),
                "actions": summary.get("actions", []),
            })
        return compact

    def get_event_details(self, event_id: str, date: Optional[str] = None) -> dict:
        return self._find_event(event_id, date)

    async def send_thumbnail(self, event_id: str, date: Optional[str] = None) -> str:
        ev = self._find_event(event_id, date)
        raw = ev.get("thumbnail")
        if not raw:
            raise EventToolError(f"no thumbnail for {event_id}")
        path = self._sandbox(raw)
        await self.telegram.send_photo(path, caption=f"Thumbnail: {event_id}")
        return f"thumbnail for {event_id} sent"

    async def send_video(self, event_id: str, date: Optional[str] = None) -> str:
        ev = self._find_event(event_id, date)
        raw = ev.get("video")
        if not raw:
            raise EventToolError(f"no video for {event_id}")
        path = self._sandbox(raw)
        await self.telegram.send_video(path, caption=f"Video: {event_id}")
        return f"video for {event_id} sent"
