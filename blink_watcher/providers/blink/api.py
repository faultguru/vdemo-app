"""Blink's clip list and clip download, over blinkpy, as ProviderClips."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from blinkpy import api as blink_api
from blinkpy.blinkpy import Blink

from blink_watcher.providers.base import LIVE_VIEW, MOTION, SNAPSHOT, ProviderClip, ProviderError
from blink_watcher.settings.cameras import camera_key

log = logging.getLogger(__name__)

PROVIDER = "blink"

# Blink's clip `source` values -> ProviderClip.trigger.
TRIGGERS = {"cv_motion": MOTION, "pir": MOTION, "snapshot": SNAPSHOT, "liveview": LIVE_VIEW}


def _clip_metadata(media: dict) -> dict:
    """Blink's `metadata` field decoded — it arrives as a JSON string, e.g.
    '{"pts_length_ms":30028,"cv_detection":["person"]}'."""
    md = media.get("metadata")
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except json.JSONDecodeError:
            return {}
    return md if isinstance(md, dict) else {}


def to_clip(media: dict, recorded_at: datetime) -> ProviderClip:
    """An entry of Blink's video list -> ProviderClip."""
    md = _clip_metadata(media)
    length_ms = md.get("pts_length_ms")
    source = media.get("source")
    return ProviderClip(
        provider=PROVIDER,
        provider_clip_id=str(media.get("id")),
        camera=media["device_name"].strip(),
        recorded_at=recorded_at,
        duration_sec=round(length_ms / 1000, 3) if isinstance(length_ms, (int, float)) else None,
        trigger=TRIGGERS.get(source, source),
        detections=list(md.get("cv_detection") or []),
        raw=media,
    )


async def list_clips(blink: Blink, cameras: Optional[list[str]], since: Optional[datetime] = None,
                     until: Optional[datetime] = None) -> AsyncIterator[ProviderClip]:
    """Clips from Blink's video list (newest first, page by page), filtered to
    `cameras` (None = every camera) and [since, until)."""
    wanted = None if cameras is None else {camera_key(c) for c in cameras}
    page = 1
    while True:
        resp = await blink_api.request_videos(blink, time=0, page=page)
        videos = (resp or {}).get("media") or []
        if not videos:
            return

        stop = False
        for v in videos:
            name = v.get("device_name")
            if not name or (wanted is not None and camera_key(name) not in wanted):
                continue
            ts_raw = v.get("created_at")
            if not ts_raw:
                continue
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
            if until is not None and ts >= until:
                continue
            if since is not None and ts < since:
                stop = True
                continue
            yield to_clip(v, ts)

        if stop:
            return
        page += 1


async def iter_clip_bytes(blink: Blink, clip: ProviderClip) -> AsyncIterator[bytes]:
    """The clip's .mp4 from Blink, in 64 KB chunks."""
    url = f"{blink.urls.base_url}{clip.raw.get('media')}"
    resp = await blink.auth.query(url=url, headers=blink.auth.header, json_resp=False, timeout=60)
    if resp is None:
        raise ProviderError(f"Blink clip download returned nothing for {url}")
    async for chunk in resp.content.iter_chunked(64 * 1024):
        yield chunk
