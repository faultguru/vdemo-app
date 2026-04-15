import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from blinkpy.blinkpy import Blink

from .blink_api import Clip, download_clip
from .config import Settings
from .store import Store

log = logging.getLogger(__name__)


class Downloader:
    def __init__(self, settings: Settings, store: Store, blink: Blink):
        self.settings = settings
        self.store = store
        self.blink = blink
        self.tz = ZoneInfo(settings.timezone)
        self._stop = asyncio.Event()
        self._idle_sleep = 5.0
        self._workers = max(1, settings.max_concurrent_downloads)

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        log.info("downloader started (%d workers)", self._workers)
        workers = [asyncio.create_task(self._worker(i)) for i in range(self._workers)]
        try:
            await self._stop.wait()
        finally:
            for w in workers:
                w.cancel()
            for w in workers:
                try:
                    await w
                except (asyncio.CancelledError, Exception):
                    pass

    async def _worker(self, idx: int) -> None:
        while not self._stop.is_set():
            if not self._has_disk_budget():
                log.warning("disk budget exhausted (%.1f GB cap) — pausing", self.settings.max_disk_gb)
                await self._sleep(60)
                continue
            with self.store.tx():
                row = self.store.claim_next_pending()
            if row is None:
                await self._sleep(self._idle_sleep)
                continue
            await self._handle(row)

    async def _handle(self, row) -> None:
        clip = Clip(
            id=row["id"],
            camera=row["camera"],
            ts=datetime.fromisoformat(row["ts"]),
            source_url=row["source_url"] or "",
        )
        dest = self._dest_path(clip)
        try:
            await download_clip(self.blink, clip, dest)
            with self.store.tx():
                self.store.mark_downloaded(clip.id, str(dest))
                self.store.mark_processed(clip.id)
            log.info("downloaded %s %s → %s", clip.camera, clip.ts.isoformat(), dest)
        except Exception as exc:
            log.exception("download failed for %s: %s", clip.id, exc)
            with self.store.tx():
                self.store.mark_failed(clip.id, str(exc))

    def _dest_path(self, clip: Clip) -> Path:
        local_ts = clip.ts.astimezone(self.tz)
        day = local_ts.strftime("%Y-%m-%d")
        stamp = local_ts.strftime("%Y-%m-%d-%H-%M-%S")
        safe_cam = clip.camera.strip().replace("/", "_").replace(" ", "_")
        return self.settings.video_dir / day / safe_cam / f"{stamp}.mp4"

    def _has_disk_budget(self) -> bool:
        budget_bytes = int(self.settings.max_disk_gb * (1024**3))
        used = _dir_size(self.settings.video_dir)
        return used < budget_bytes

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total
