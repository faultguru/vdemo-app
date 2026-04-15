import asyncio
import logging
from datetime import datetime, timezone

from blinkpy.blinkpy import Blink

from .blink_api import list_clips
from .config import Settings
from .store import Store

log = logging.getLogger(__name__)


class LivePoller:
    def __init__(self, settings: Settings, store: Store, blink: Blink):
        self.settings = settings
        self.store = store
        self.blink = blink
        self._stop = asyncio.Event()
        self._started_at = datetime.now(timezone.utc)

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        log.info("live poller started; watching %s", self.settings.cameras)
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception as exc:
                log.exception("poll tick failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.poll_interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        await self.blink.refresh(force=True)
        new_count = 0
        async for clip in list_clips(
            self.blink, self.settings.cameras, since=self._started_at
        ):
            with self.store.tx():
                if self.store.enqueue(clip.id, clip.camera, clip.ts, clip.source_url, priority=0):
                    new_count += 1
        if new_count:
            log.info("enqueued %d live clips", new_count)
