"""What every route of the web UI shares: the settings, the local time zone,
the storage areas it may serve files from, and the camera list."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from aiohttp import web

from blink_watcher.core.storage import (
    IGNORED_VIDEOS, LIVE_SNAPSHOTS, PROCESSED_VIDEOS, RAW_VIDEOS, TEMP_VIDEOS, area_config,
)
from blink_watcher.settings import Settings
from blink_watcher.stitch.stitcher import Stitcher
from blink_watcher.webui.cameras import camera_list

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Areas whose files (videos, pictures, .metadata.json) the UI may serve.
MEDIA_AREAS = (RAW_VIDEOS, PROCESSED_VIDEOS, IGNORED_VIDEOS, TEMP_VIDEOS, LIVE_SNAPSHOTS)


@dataclass
class WebContext:
    settings: Settings
    root: Path = field(init=False)                  # STORAGE_ROOT; event paths are relative to it
    chat_dir: Path = field(init=False)
    tz: ZoneInfo = field(init=False)
    areas: set[str] = field(init=False)             # folder names of MEDIA_AREAS
    stitcher: Optional[Stitcher] = field(init=False)

    def __post_init__(self) -> None:
        s = self.settings
        self.root = s.storage_path
        self.chat_dir = s.chat_dir
        self.tz = ZoneInfo(s.timezone)
        self.areas = {area_config(s, a)[1] for a in MEDIA_AREAS}
        self.stitcher = Stitcher(s) if s.stitch_enabled else None

    def today(self) -> str:
        return datetime.now(self.tz).strftime("%Y-%m-%d")

    def now_iso(self) -> str:
        return datetime.now(self.tz).isoformat(timespec="seconds")

    def cameras(self) -> list[dict]:
        """Every known camera with its CAMERAS (enabled) and CAMERA_FILTERS (filter) settings."""
        return camera_list(self.settings)

    def resolve(self, rel: str, suffixes: set[str]) -> Path:
        """`rel` under STORAGE_ROOT, inside a media area, with an allowed suffix — else 404."""
        path = (self.root / rel).resolve()
        try:
            parts = path.relative_to(self.root).parts
        except ValueError:
            raise web.HTTPNotFound()
        if not parts or parts[0] not in self.areas or not any(path.name.endswith(s) for s in suffixes):
            raise web.HTTPNotFound()
        if not path.is_file():
            raise web.HTTPNotFound()
        return path


CONTEXT = web.AppKey("context", WebContext)


def context(request: web.Request) -> WebContext:
    return request.app[CONTEXT]


def date_param(request: web.Request) -> str:
    """The route's {date}, checked: YYYY-MM-DD, else 400."""
    date = request.match_info["date"]
    if not DATE.match(date):
        raise web.HTTPBadRequest(text="date must be YYYY-MM-DD")
    return date
