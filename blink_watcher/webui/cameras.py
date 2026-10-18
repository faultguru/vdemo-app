"""The one camera list of the web UI (console filters, configuration page):
the camera catalog (providers.catalog, written whenever a provider session
lists the account's cameras). Until a catalog exists, the cameras named in
CAMERAS plus those seen in the last FALLBACK_DAYS days' events stand in."""
from __future__ import annotations

from blink_watcher.chat import store as chat_store
from blink_watcher.providers.base import Camera
from blink_watcher.providers.catalog import load_cameras
from blink_watcher.settings import Settings
from blink_watcher.settings.cameras import camera_key

FALLBACK_DAYS = 14


def _fallback(settings: Settings) -> list[Camera]:
    names = [] if settings.all_cameras else [c.strip() for c in settings.cameras]
    for date in sorted(chat_store.event_dates(settings.chat_dir), reverse=True)[:FALLBACK_DAYS]:
        names += [e["camera"] for e in chat_store.load_events(settings.chat_dir, date) if e.get("camera")]
    unique = {camera_key(n): n for n in reversed(names)}  # the first spelling wins
    return [Camera(name=n, provider=settings.camera_provider, id="") for n in unique.values()]


def camera_list(settings: Settings) -> list[dict]:
    """[{name, provider, id, model, ..., in_catalog, enabled, filter, indoor}], by name."""
    cameras = load_cameras(settings)
    in_catalog = bool(cameras)
    cameras = cameras or _fallback(settings)
    return [{**c.to_dict(), "in_catalog": in_catalog, "enabled": settings.camera_enabled(c.name),
             "filter": settings.camera_filter(c.name), "indoor": settings.camera_indoor(c.name)}
            for c in sorted(cameras, key=lambda c: c.name.lower())]
