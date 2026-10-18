"""One camera's live view as JPEG frames: the provider's LiveSource -> ffmpeg
(the imageio-ffmpeg binary, CPU only) -> frames for every viewer, served by
the web UI as MJPEG. A stream ends when the provider ends it, when no frame
came for LIVE_STALL_SEC, or LIVE_IDLE_SEC after its last viewer left."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

from blink_watcher.live.snapshots import SOI
from blink_watcher.providers import LiveSource, ProviderError

if TYPE_CHECKING:
    from blink_watcher.live.monitor import LiveMonitor

log = logging.getLogger(__name__)

LIVE_FPS = 8                # frames per second of the MJPEG stream
LIVE_SNAPSHOT_SEC = 5.0     # a live frame saved as the camera's snapshot this often
LIVE_IDLE_SEC = 8.0         # stream stops this long after its last viewer left
LIVE_STALL_SEC = 25.0       # no frame for this long: the stream is over
EOI = b"\xff\xd9"           # JPEG end-of-image marker


class LiveError(RuntimeError):
    pass


class LiveStream:
    """Shared by every viewer of the camera: viewers wait on next_frame(); the
    web UI counts them in `viewers` and sets `left_at` when one leaves."""

    def __init__(self, monitor: "LiveMonitor", name: str):
        self.monitor = monitor
        self.name = name
        self.frame: Optional[bytes] = None
        self.seq = 0
        self.viewers = 0
        self.left_at = time.monotonic()
        self.done = False
        self.error: Optional[str] = None
        self.started = asyncio.Event()      # first frame arrived, or the stream failed
        self._cond = asyncio.Condition()
        self.task = asyncio.create_task(self._run())

    async def next_frame(self, after: int) -> tuple[Optional[bytes], int]:
        """The first frame newer than `after` -> (jpeg, seq); (None, seq) once the stream is over."""
        async with self._cond:
            await self._cond.wait_for(lambda: self.seq > after or self.done)
            return (None if self.seq <= after else self.frame), self.seq

    async def _publish(self, frame: Optional[bytes] = None, done: bool = False) -> None:
        async with self._cond:
            if frame is not None:
                self.frame, self.seq = frame, self.seq + 1
            self.done = self.done or done
            self._cond.notify_all()
        self.started.set()

    async def _run(self) -> None:
        import imageio_ffmpeg

        proc = None
        source: Optional[LiveSource] = None
        t0 = time.perf_counter()
        try:
            provider = await self.monitor.connect()
            log.info(f"Live view requested: {self.name}")
            source = await provider.live_stream(self.name)
            proc = await asyncio.create_subprocess_exec(
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                "-fflags", "nobuffer", "-flags", "low_delay", *source.input_args,
                "-an", "-vf", f"fps={LIVE_FPS}", "-q:v", "5", "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1",
                stdin=asyncio.subprocess.PIPE if source.uses_stdin else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await source.start(proc.stdin if source.uses_stdin else None)
            log.info(f"Live view started: {self.name} ({source.kind} stream, {time.perf_counter() - t0:.1f}s)")
            await self._pump(proc)
        except Exception as e:  # noqa: BLE001
            self.error = str(e) if isinstance(e, (LiveError, ProviderError)) else f"{type(e).__name__}: {e}"
            log.warning(f"Live view failed: {self.name} - {self.error}")
        finally:
            if source:
                await source.stop()
            if proc and proc.returncode is None:
                proc.kill()
            if proc:
                err = (await proc.stderr.read()).decode("utf-8", "replace").strip()
                if err and not self.seq:
                    self.error = self.error or f"ffmpeg: {err.splitlines()[-1]}"
                    log.warning(f"Live view ffmpeg error: {self.name} - {err[-400:]}")
            await self._publish(done=True)
            self.monitor.stream_ended(self)
            log.info(f"Live view stopped: {self.name} ({self.seq} frames, {time.perf_counter() - t0:.0f}s)")

    async def _pump(self, proc: asyncio.subprocess.Process) -> None:
        """ffmpeg's JPEG frames -> viewers (+ a snapshot now and then), until
        the provider ends the stream, it stalls, or nobody has watched for a while."""
        buf, saved = b"", 0.0
        while True:
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(256 * 1024), LIVE_STALL_SEC)
            except asyncio.TimeoutError:
                log.info(f"Live view stalled: {self.name} (no data for {LIVE_STALL_SEC:g}s)")
                return
            if not chunk:
                return
            buf += chunk
            while (start := buf.find(SOI)) >= 0 and (end := buf.find(EOI, start + 2)) >= 0:
                frame, buf = buf[start:end + 2], buf[end + 2:]
                await self._publish(frame)
                if time.monotonic() - saved >= LIVE_SNAPSHOT_SEC:
                    saved = time.monotonic()
                    await self.monitor.save_snapshot(self.name, frame, source="live")
            if self.viewers == 0 and time.monotonic() - self.left_at > LIVE_IDLE_SEC:
                return
