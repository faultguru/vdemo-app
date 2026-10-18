"""The camera catalog: the provider account's cameras, kept in STORAGE_ROOT/state/cameras.json.

Written whenever a provider session lists its cameras (refresh(): the fetch
daemon at start and hourly, the live monitor on connect), so the web UI and
the configuration page know the cameras without a provider session of their
own. File:

  {"updated_at": "2026-10-18T09:00:00-07:00",
   "cameras": [{"name": "Front Door", "provider": "blink", "id": "123", "model": "catalina",
                "has_own_person_detection": null}, ...]}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from blink_watcher.core.runtime import state_dir
from blink_watcher.providers.base import Camera, CameraProvider
from blink_watcher.settings import Settings

log = logging.getLogger(__name__)


def catalog_file(settings: Settings) -> Path:
    return state_dir(settings) / "cameras.json"


def load_cameras(settings: Settings) -> list[Camera]:
    """The catalog's cameras; [] if no provider session has listed them yet."""
    path = catalog_file(settings)
    if not path.exists():
        return []
    try:
        return [Camera.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8")).get("cameras") or []]
    except (ValueError, KeyError, AttributeError) as e:
        log.warning("Camera catalog unreadable: %s (%s)", path, e)
        return []


def save_cameras(settings: Settings, cameras: list[Camera]) -> None:
    path = catalog_file(settings)
    data = {"updated_at": datetime.now(ZoneInfo(settings.timezone)).isoformat(timespec="seconds"),
            "cameras": [c.to_dict() for c in sorted(cameras, key=lambda c: c.name.lower())]}
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    part.replace(path)


async def refresh(settings: Settings, provider: CameraProvider) -> list[Camera]:
    """The provider's cameras, saved as the catalog."""
    cameras = await provider.list_cameras()
    save_cameras(settings, cameras)
    log.info("Camera catalog updated: %d camera(s)", len(cameras))
    return cameras
