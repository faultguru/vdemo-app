"""Thin wrapper over blinkpy so the rest of the app doesn't touch its internals."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

from blinkpy import api as blink_api
from blinkpy.blinkpy import Blink

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Clip:
    id: str
    camera: str
    ts: datetime
    source_url: str


async def list_clips(
    blink: Blink,
    cameras: list[str],
    since: datetime | None = None,
    until: datetime | None = None,
) -> AsyncIterator[Clip]:
    """Yield clips from Blink's video list API, filtered to `cameras` and [since, until)."""
    page = 1
    while True:
        resp = await blink_api.request_videos(blink, time=0, page=page)
        videos = (resp or {}).get("media") or []
        if not videos:
            return

        stop = False
        for v in videos:
            camera_name = v.get("device_name")
            if camera_name not in cameras:
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

            yield Clip(
                id=str(v.get("id")),
                camera=camera_name,
                ts=ts,
                source_url=v.get("media"),
            )

        if stop:
            return
        page += 1


async def download_clip(blink: Blink, clip: Clip, dest: "Path") -> None:  # type: ignore[name-defined]
    import aiofiles
    url = f"{blink.urls.base_url}{clip.source_url}"
    resp = await blink.auth.query(
        url=url, headers=blink.auth.header, json_resp=False, timeout=60
    )
    if resp is None:
        raise RuntimeError(f"download request returned None for {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(dest, "wb") as f:
        async for chunk in resp.content.iter_chunked(64 * 1024):
            await f.write(chunk)
