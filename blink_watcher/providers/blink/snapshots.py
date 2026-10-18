"""Blink snapshots: ask cameras for a new picture, then fetch the new thumbnails."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional
from urllib.parse import urljoin

from blinkpy import api
from blinkpy.blinkpy import Blink
from blinkpy.camera import BlinkCamera

from blink_watcher.providers.base import Snapshot, find_camera

log = logging.getLogger(__name__)


async def take_snapshots(blink: Blink, names: list[str]) -> dict[str, Snapshot]:
    """Ask every camera for a new picture, wait until they are taken, then
    fetch the new thumbnails (Blink spaces camera commands ~2 s apart)."""
    out: dict[str, Snapshot] = {}
    commands: list[tuple[str, BlinkCamera, object]] = []
    for name in names:
        camera = find_camera(blink.cameras, name)
        if camera is None:
            out[name] = Snapshot(name, error="camera not found on the Blink account")
            continue
        try:
            resp = await api.request_new_image(blink, camera.network_id, camera.camera_id,
                                               camera_type=camera.camera_type)
        except Exception as e:  # noqa: BLE001
            out[name] = Snapshot(name, error=f"snapshot request failed: {e}")
            continue
        commands.append((name, camera, resp))
    await asyncio.gather(*(api.wait_for_command(blink, r) for _, _, r in commands if isinstance(r, dict)),
                         return_exceptions=True)
    await blink.get_homescreen()
    thumbs = {}
    for kind in ("cameras", "owls", "doorbells"):
        for item in (blink.homescreen or {}).get(kind) or []:
            thumbs[str(item.get("id"))] = item.get("thumbnail")
    for name, camera, _ in commands:
        url = thumbnail_url(blink, camera, thumbs.get(str(camera.camera_id)))
        if not url:
            out[name] = Snapshot(name, error="no thumbnail from Blink")
            continue
        resp = await camera.get_thumbnail(url)
        if not resp or resp.status != 200:
            out[name] = Snapshot(name, error=f"thumbnail download failed ({getattr(resp, 'status', 'no response')})")
            continue
        out[name] = Snapshot(name, data=await resp.read())
    return out


def thumbnail_url(blink: Blink, camera: BlinkCamera, thumb) -> Optional[str]:
    """The homescreen's thumbnail value -> its URL (as blinkpy's camera.update_images builds it)."""
    if not thumb:
        return None
    thumb = str(thumb)
    if thumb.isdigit():  # newer API: only the timestamp
        path = (f"/api/v3/media/accounts/{blink.account_id}/networks/{camera.network_id}/"
                f"{camera.product_type}/{camera.camera_id}/thumbnail/thumbnail.jpg?ts={thumb}&ext=")
    else:
        path = thumb if thumb.endswith("&ext=") else f"{thumb}.jpg"
    return urljoin(camera.sync.urls.base_url, path)
