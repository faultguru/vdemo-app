"""Live monitoring for the web UI: fresh camera snapshots and true live view, through the camera provider.

Snapshots: while a browser shows the Live monitor it calls status() every few
seconds (the heartbeat, with the cameras it shows). As long as heartbeats
come, a loop asks the provider for a new picture of each of those cameras,
round after round, as often as the provider allows (Blink: one camera command
every ~2 s, then the picture is ready a few seconds later). Every new picture
is saved in the LIVE_SNAPSHOTS area (live.snapshots) and kept
WEBUI_LIVE_KEEP_HOURS, so the AI analyst can look at the latest ones
(ai_context()). The loop stops when the heartbeats stop.

Live view: stream(name) -> the camera's shared LiveStream (live.stream). While
it runs, a frame every LIVE_SNAPSHOT_SEC is saved as the camera's snapshot
(the snapshot loop skips the camera meanwhile).

The provider session is opened on first use; connecting also refreshes the
camera catalog (providers.catalog).
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from blink_watcher.ai import ImagePart, Part, TextPart
from blink_watcher.core.images import ImageError, fit_jpeg
from blink_watcher.live.snapshots import SnapshotStore
from blink_watcher.live.stream import LiveError, LiveStream
from blink_watcher.providers import CameraProvider, catalog, make_provider
from blink_watcher.settings import Settings

log = logging.getLogger(__name__)

HEARTBEAT_SEC = 15.0        # snapshots continue this long after the last status() call
ROUND_PAUSE_SEC = 1.0       # pause between snapshot rounds
ERROR_PAUSE_SEC = 10.0      # pause after a failed round
PRUNE_EVERY_SEC = 600.0


class LiveMonitor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tz = ZoneInfo(settings.timezone)
        self.snapshots = SnapshotStore(settings)
        self.provider: Optional[CameraProvider] = None
        self.camera_names: list[str] = []            # the provider account's cameras
        self.connect_error: Optional[str] = None
        self._connect_lock = asyncio.Lock()
        self.watched: dict[str, float] = {}          # camera -> last heartbeat (monotonic)
        self.latest: dict[str, dict] = {}            # camera -> {path, at, source}
        self.errors: dict[str, str] = {}
        self.streams: dict[str, LiveStream] = {}
        self._loop: Optional[asyncio.Task] = None
        self._pruned = 0.0

    # ---------- provider ----------

    async def connect(self) -> CameraProvider:
        async with self._connect_lock:
            if self.provider is None:
                provider = make_provider(self.settings)
                log.info(f"Connecting to {provider.name}...")
                try:
                    await provider.connect()
                    cameras = await catalog.refresh(self.settings, provider)
                except Exception as e:  # noqa: BLE001
                    await provider.close()
                    self.connect_error = str(e)
                    log.error(f"Connection to {provider.name} failed: {e}")
                    raise LiveError(f"{provider.name} connection failed: {e}") from e
                self.provider, self.connect_error = provider, None
                self.camera_names = [c.name for c in cameras]
                log.info(f"Connected to {provider.name}: {len(cameras)} camera(s)")
            return self.provider

    async def cameras(self) -> list[str]:
        """The enabled cameras (CAMERAS) on the provider account."""
        await self.connect()
        return sorted((n for n in self.camera_names if self.settings.camera_enabled(n)), key=str.lower)

    async def close(self) -> None:
        if self._loop:
            self._loop.cancel()
        for s in list(self.streams.values()):
            s.task.cancel()
        if self.provider is not None:
            await self.provider.close()

    # ---------- snapshots ----------

    def status(self, cams: list[str]) -> dict:
        """Heartbeat of a browser showing `cams` -> their latest snapshots."""
        now = time.monotonic()
        for c in cams:
            self.watched[c] = now
        if cams and (self._loop is None or self._loop.done()):
            self._loop = asyncio.create_task(self._snap_loop())
        out = {}
        for c in cams:
            snap = self.latest.get(c) or self.snapshots.latest(c)
            if snap:
                self.latest.setdefault(c, snap)
            s = self.streams.get(c)
            out[c] = {**(snap or {}), "live": bool(s and not s.done), "error": self.errors.get(c)}
        return {"cameras": out, "connected": self.provider is not None, "error": self.connect_error}

    def _watched_now(self) -> list[str]:
        now = time.monotonic()
        return [c for c, t in self.watched.items() if now - t < HEARTBEAT_SEC]

    async def _snap_loop(self) -> None:
        log.info("Snapshot loop started")
        rounds = 0
        try:
            while cams := self._watched_now():
                cams = [c for c in cams if c not in self.streams]
                if not cams:
                    await asyncio.sleep(ROUND_PAUSE_SEC)
                    continue
                try:
                    await self._snap_round(cams)
                    rounds += 1
                except LiveError:
                    await asyncio.sleep(ERROR_PAUSE_SEC)
                except Exception:  # noqa: BLE001
                    log.exception("Snapshot round failed")
                    await asyncio.sleep(ERROR_PAUSE_SEC)
                await asyncio.sleep(ROUND_PAUSE_SEC)
                if time.monotonic() - self._pruned > PRUNE_EVERY_SEC:
                    self._pruned = time.monotonic()
                    await asyncio.to_thread(self.snapshots.prune)
        finally:
            log.info(f"Snapshot loop stopped ({rounds} round(s); no viewer)")

    async def _snap_round(self, cams: list[str]) -> None:
        provider = await self.connect()
        t0 = time.perf_counter()
        fresh = 0
        for name, snap in (await provider.snapshots(cams)).items():
            if snap.data is None:
                self.errors[name] = snap.error or "no picture"
                continue
            if await self.save_snapshot(name, snap.data, source="snapshot"):
                fresh += 1
            self.errors.pop(name, None)
        log.debug(f"Snapshot round: {fresh}/{len(cams)} new picture(s) in {time.perf_counter() - t0:.1f}s")

    async def save_snapshot(self, name: str, data: bytes, source: str) -> bool:
        """Store a picture as the camera's latest snapshot; False if it is not a new picture."""
        snap = await asyncio.to_thread(self.snapshots.save, name, data, source)
        if snap:
            self.latest[name] = snap
        return snap is not None

    # ---------- live view ----------

    def stream(self, name: str) -> LiveStream:
        s = self.streams.get(name)
        if s is None or s.done:
            s = self.streams[name] = LiveStream(self, name)
        return s

    def stream_ended(self, stream: LiveStream) -> None:
        if self.streams.get(stream.name) is stream:
            del self.streams[stream.name]

    # ---------- AI analyst ----------

    def ai_context(self, cams: list[str], frames: int, window_sec: float) -> tuple[str, list[Part]]:
        """What the analyst gets about the live monitor: a text line per camera
        and its latest snapshots (up to `frames`, spread over the last
        `window_sec`, oldest first), each a TextPart label + an ImagePart."""
        now = datetime.now(self.tz)
        lines: list[str] = []
        parts: list[Part] = []
        for name in cams:
            files = self.snapshots.files(name, since=now - timedelta(seconds=window_sec)) \
                or self.snapshots.files(name)[-1:]
            if not files:
                lines.append(f"- {name}: no snapshot yet")
                continue
            if len(files) > frames:  # the newest + evenly spaced earlier ones
                step = (len(files) - 1) / max(1, frames - 1)
                files = [files[round(i * step)] for i in range(frames)]
            ages = ", ".join(f"{ts:%H:%M:%S} ({int((now - ts).total_seconds())}s ago)" for ts, _ in files)
            lines.append(f"- {name}: {len(files)} snapshot(s) at {ages}")
            for ts, key in files:
                try:
                    image = ImagePart(fit_jpeg(self.snapshots.read(key)))
                except (OSError, ImageError):
                    continue
                parts += [TextPart(f"Live snapshot - {name} at {ts:%H:%M:%S} "
                                   f"({int((now - ts).total_seconds())}s ago):"), image]
        return "\n".join(lines), parts
