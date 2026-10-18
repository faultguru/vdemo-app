"""The live monitor's stored snapshots, in the LIVE_SNAPSHOTS storage area
(LIVE_SNAPSHOTS_STORAGE_TYPE / LIVE_SNAPSHOTS_PREFIX, default STORAGE_ROOT/videos.live):

  <YYYY-MM-DD>/<Camera_Name>/<YYYY-MM-DD-HH-MM-SS>.jpg     local time

kept WEBUI_LIVE_KEEP_HOURS (prune()). A snapshot is described to the web UI as
{"path": <file relative to STORAGE_ROOT>, "at": <local ISO time>, "source": ...}.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Optional
from zoneinfo import ZoneInfo

from blink_watcher.core.storage import LIVE_SNAPSHOTS, camera_folder, make_storage
from blink_watcher.settings import Settings

log = logging.getLogger(__name__)

SOI = b"\xff\xd8"   # JPEG start-of-image marker
STAMP = "%Y-%m-%d-%H-%M-%S"


class SnapshotStore:
    def __init__(self, settings: Settings):
        self.storage = make_storage(settings, LIVE_SNAPSHOTS)
        self.tz = ZoneInfo(settings.timezone)
        self.keep_hours = settings.webui_live_keep_hours
        self._digest: dict[str, int] = {}   # camera -> hash of its last picture

    def link(self, key: str, at: datetime, source: str) -> dict:
        return {"path": f"{self.storage.folder}/{key}", "at": at.isoformat(timespec="seconds"), "source": source}

    def save(self, name: str, data: bytes, source: str) -> Optional[dict]:
        """Store a picture as the camera's latest snapshot -> its link; None if it is
        the same picture as last time (the provider had no new one) or not a JPEG."""
        if not data.startswith(SOI) or hash(data) == self._digest.get(name):
            return None
        self._digest[name] = hash(data)
        now = datetime.now(self.tz)
        key = f"{now:%Y-%m-%d}/{camera_folder(name)}/{now:{STAMP}}.jpg"
        self.storage.write_bytes(key, data)
        return self.link(key, now, source)

    def files(self, name: str, since: Optional[datetime] = None) -> list[tuple[datetime, str]]:
        """(time, key) of the camera's stored snapshots, oldest first; today's
        and yesterday's folders (enough for any window that crosses midnight)."""
        today = datetime.now(self.tz)
        out = []
        for day in (today - timedelta(days=1), today):
            for key in self.storage.list(f"{day:%Y-%m-%d}/{camera_folder(name)}"):
                ts = self._time(key)
                if ts is not None and (since is None or ts >= since):
                    out.append((ts, key))
        return sorted(out)

    def latest(self, name: str) -> Optional[dict]:
        files = self.files(name)
        return self.link(files[-1][1], files[-1][0], "stored") if files else None

    def read(self, key: str) -> bytes:
        return self.storage.read_bytes(key)

    def prune(self) -> None:
        """Delete snapshots older than WEBUI_LIVE_KEEP_HOURS, and emptied day folders."""
        cutoff = datetime.now(self.tz) - timedelta(hours=self.keep_hours)
        removed, days, kept_days = 0, set(), set()
        for key in list(self.storage.list()):
            day = PurePosixPath(key).parts[0]
            days.add(day)
            ts = self._time(key)
            if ts is not None and ts < cutoff:
                self.storage.delete(key)
                removed += 1
            else:
                kept_days.add(day)
        for day in days - kept_days:
            self.storage.delete_tree(day)
        if removed:
            log.info(f"Old snapshots deleted: {removed} (older than {self.keep_hours:g} h)")

    def _time(self, key: str) -> Optional[datetime]:
        p = PurePosixPath(key)
        if p.suffix != ".jpg":
            return None
        try:
            return datetime.strptime(p.stem, STAMP).replace(tzinfo=self.tz)
        except ValueError:
            return None
