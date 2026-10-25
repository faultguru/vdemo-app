"""The chat agent's tools on the recorded events: list, search and read
them, look at their pictures, send their thumbnail or video to the user.

Events are read from the chat area's day indexes (chat/<date>/events.json,
see chat.store). File paths in an event are relative to STORAGE_ROOT
(e.g. videos.processed/...) and sandboxed: anything resolving outside
STORAGE_ROOT is rejected.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional, Protocol

from blink_watcher.ai import ImagePart
from blink_watcher.chat import store as chat_store
from blink_watcher.core.images import fit_jpeg
from blink_watcher.settings import Settings

log = logging.getLogger(__name__)


class MediaSink(Protocol):
    """Where send_thumbnail / send_video deliver: the Telegram client, or the
    web console's collector."""

    async def send_photo(self, photo: Path, caption: Optional[str] = None) -> dict: ...

    async def send_video(self, video: Path, caption: Optional[str] = None) -> dict: ...


class EventToolError(Exception):
    pass


class EventTools:
    def __init__(self, settings: Settings, telegram: Optional[MediaSink]):
        """`telegram`: where send_thumbnail / send_video deliver (None: the
        notifier, whose role has no tools)."""
        self.settings = settings
        self.telegram = telegram
        self._chat_dir = settings.chat_dir
        self._storage_root = settings.storage_path

    # ---------- sandbox helpers ----------

    def _sandbox(self, raw: str) -> Path:
        candidate = (self._storage_root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        try:
            candidate.relative_to(self._storage_root)
        except ValueError as e:
            raise EventToolError(f"path escapes storage sandbox: {raw}") from e
        if not candidate.exists():
            raise EventToolError(f"file not found: {raw}")
        return candidate

    def _load_date(self, date: str) -> list[dict]:
        if date not in self.list_event_dates():
            return []  # nothing recorded that day
        return chat_store.load_events(self._chat_dir, date)

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
        return chat_store.event_dates(self._chat_dir)

    def list_events(self, date: str) -> list[dict]:
        return [self._compact(ev) for ev in self._load_date(date)]

    @staticmethod
    def _compact(ev: dict) -> dict:
        summary = ev.get("summary") or {}
        return {
            "event_id": ev.get("event_id"),
            "camera": ev.get("camera"),
            "recorded_at": ev.get("recorded_at"),
            "description": summary.get("description", ""),
            "people_types": summary.get("people_types", []),
            "labels": summary.get("labels", []),  # the describer's free-form labels
            "threat": summary.get("threat"),      # {level, signals, evidence}; None = nothing suspicious
            "actions": summary.get("actions", []),
            "track": ev.get("track"),  # same subject as earlier events (camera-to-camera stitching)
            "notification": {k: v for k, v in (ev.get("notification") or {}).items()
                             if k in ("decision", "message", "skip_reason", "sent_at")} or None,
        }

    def get_event_details(self, event_id: str, date: Optional[str] = None) -> dict:
        return self._find_event(event_id, date)

    def search_events(self, text: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                      camera: Optional[str] = None, limit: int = 30) -> list[dict]:
        """Events in [start_date, end_date] (all dates if open) whose text holds
        any of the words in `text`, best matches first (then newest)."""
        words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1]
        if not words:
            raise EventToolError("search text has no words")
        dates = [d for d in self.list_event_dates()
                 if (not start_date or d >= start_date) and (not end_date or d <= end_date)]
        scored = []
        for d in dates:
            for ev in chat_store.load_events(self._chat_dir, d):
                if camera and (ev.get("camera") or "").strip().lower() != camera.strip().lower():
                    continue
                hay = json.dumps({k: ev.get(k) for k in ("camera", "summary", "track", "notification")},
                                 ensure_ascii=False).lower()
                score = sum(hay.count(w) > 0 for w in words)
                if score:
                    scored.append((score, ev.get("recorded_at") or "", ev))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [{**self._compact(ev), "matched_words": score} for score, _, ev in scored[:limit]]

    def event_pictures(self, event_ids: list[str]) -> list[tuple[str, str]]:
        """(event_id, thumbnail path relative to STORAGE_ROOT) of the events that have one."""
        out = []
        for event_id in event_ids:
            try:
                raw = self._find_event(event_id).get("thumbnail")
            except EventToolError:
                continue
            if raw:
                out.append((event_id, raw))
        if not out:
            raise EventToolError("none of these events has a picture")
        return out

    def picture(self, raw: str) -> ImagePart:
        """A picture under STORAGE_ROOT, downscaled for the model."""
        return ImagePart(fit_jpeg(self._sandbox(raw).read_bytes()))

    async def send_thumbnail(self, event_id: str, date: Optional[str] = None) -> str:
        ev = self._find_event(event_id, date)
        raw = ev.get("thumbnail")
        if not raw:
            raise EventToolError(f"no thumbnail for {event_id}")
        path = self._sandbox(raw)
        await self.telegram.send_photo(path, caption=f"Thumbnail: {event_id}")
        return f"thumbnail for {event_id} sent"

    async def send_video(self, event_id: str, date: Optional[str] = None) -> str:
        """Always the processed (trimmed) video — the event's "processed"
        link; "video" only for events without one (legacy events)."""
        ev = self._find_event(event_id, date)
        link = ev.get("processed")
        raw = f"{link['folder']}/{link['video_file']}" if link else ev.get("video")
        if not raw:
            raise EventToolError(f"no processed video for {event_id}")
        path = self._sandbox(raw)
        log.info(f"  Sending processed video {raw} for event {event_id}...")
        await self.telegram.send_video(path, caption=f"Video: {event_id}")
        return f"processed video for {event_id} sent"
