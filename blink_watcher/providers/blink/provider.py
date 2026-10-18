"""BlinkProvider: the CameraProvider for Blink (CAMERA_PROVIDER=blink)."""
from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator, Optional

from blinkpy.blinkpy import Blink

from blink_watcher.providers.base import Camera, CameraProvider, LiveSource, ProviderClip, ProviderError, Snapshot, \
    find_camera
from blink_watcher.providers.blink import api, auth
from blink_watcher.providers.blink.live import open_live_stream
from blink_watcher.providers.blink.snapshots import take_snapshots
from blink_watcher.settings import Settings


class BlinkProvider(CameraProvider):
    name = api.PROVIDER

    def __init__(self, settings: Settings):
        self.settings = settings
        self._blink: Optional[Blink] = None

    @property
    def blink(self) -> Blink:
        if self._blink is None:
            raise ProviderError("not connected to Blink")
        return self._blink

    async def connect(self) -> None:
        if self._blink is None:
            self._blink = await auth.connect(self.settings)

    async def close(self) -> None:
        if self._blink is not None:
            await self._blink.auth.session.close()
            self._blink = None

    async def list_cameras(self) -> list[Camera]:
        return [Camera(name=name.strip(), provider=self.name, id=str(cam.camera_id),
                       model=cam.product_type or cam.camera_type or None)
                for name, cam in (self.blink.cameras or {}).items()]

    def list_clips(self, cameras: Optional[list[str]], since: Optional[datetime] = None,
                   until: Optional[datetime] = None) -> AsyncIterator[ProviderClip]:
        return api.list_clips(self.blink, cameras, since, until)

    def download_clip(self, clip: ProviderClip) -> AsyncIterator[bytes]:
        return api.iter_clip_bytes(self.blink, clip)

    async def snapshots(self, cameras: list[str]) -> dict[str, Snapshot]:
        return await take_snapshots(self.blink, cameras)

    async def live_stream(self, camera: str) -> LiveSource:
        cam = find_camera(self.blink.cameras, camera)
        if cam is None:
            raise ProviderError(f"camera {camera!r} not found on the Blink account")
        return await open_live_stream(self.blink, cam)
