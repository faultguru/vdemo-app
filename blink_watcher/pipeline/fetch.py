"""Downloader service: one day's clips from the camera provider (CAMERA_PROVIDER), plus each clip's metadata.

For every clip of the chosen camera(s) on the chosen day (local time), oldest
first, writes into the raw videos storage area (RAW_VIDEOS_STORAGE_TYPE /
RAW_VIDEOS_PREFIX; on localdisk: STORAGE_ROOT/RAW_VIDEOS_PREFIX):

  <YYYY-MM-DD>/<Camera_Name>/<YYYY-MM-DD-HH-MM-SS>.mp4            the clip, as the provider serves it
  <YYYY-MM-DD>/<Camera_Name>/<YYYY-MM-DD-HH-MM-SS>.metadata.json  our provider-neutral clip record
                                                                  (format: pipeline.clip_metadata)

Each newly downloaded clip is also queued for processing: a copy of its
.metadata.json goes to the processed videos area (PROCESSED_VIDEOS_STORAGE_TYPE /
PROCESSED_VIDEOS_PREFIX) as inbox/<YYYY-MM-DD-HH-MM-SS>-<CAMERA>.metadata.json.

Clips whose .mp4 already exists are skipped (their metadata is still written
if missing), so re-running continues where the last run stopped. Clips whose
provider detections fail their camera's filter (CAMERA_FILTERS: always |
person | person_or_vehicle, default person; --filter overrides) are not
downloaded at all, to save bandwidth: only their .metadata.json is written,
marked "download": {"status": "filtered", ...}, so later runs skip them without
re-checking, and get an entry in the ignored area (IGNORED_VIDEOS:
<YYYY-MM-DD>/<Camera_Name>/<stamp>.metadata.json, "ignored": {"by": "filter"}).
Clips of cameras not enabled in CAMERAS get only their .metadata.json, marked
"download": {"status": "skipped", ...}. Neither is queued for processing.

Settings changes apply going forward: a clip that already has a record keeps
the decision taken when it was first seen, so enabling a camera or loosening
its filter never pulls in clips recorded before (--filter, a manual override,
re-checks filtered clips on purpose).

Every provider session also refreshes the camera catalog (providers.catalog).

Usage:
    python -m blink_watcher fetch                       # today, every camera in CAMERAS
    python -m blink_watcher fetch --limit 1             # just the next not-yet-downloaded clip
    python -m blink_watcher fetch --list                # list only, download nothing
    python -m blink_watcher fetch --metadata-only       # rewrite .metadata.json of stored clips; no .mp4 download
    python -m blink_watcher fetch --date 2026-10-16 --camera "Front Door" --camera "B-Doorbell"
    python -m blink_watcher fetch --daemon              # keep running: today's new clips, every POLL_INTERVAL_SECONDS
    python -m blink_watcher fetch --daemon --interval 10

--daemon keeps one provider session and re-lists today's clips (the day
follows midnight) every poll; clips already stored are skipped, so only new
ones are downloaded and queued. A failed poll is logged and retried on the
next one. The camera catalog is refreshed hourly.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Optional
from zoneinfo import ZoneInfo

from blink_watcher.core.filters import FILTER_NAMES, passes, summary
from blink_watcher.core.logs import setup_logging
from blink_watcher.core.storage import (
    IGNORED_VIDEOS, PROCESSED_VIDEOS, RAW_VIDEOS, inbox_key, make_storage, metadata_key, raw_clip_key,
)
from blink_watcher.pipeline.clip_metadata import (
    clip_metadata, decided, filtered_by, filtered_download, skipped_download,
)
from blink_watcher.providers import CameraProvider, ProviderClip, ProviderError, catalog, make_provider
from blink_watcher.settings import Settings, load_settings

log = logging.getLogger("blink_watcher.pipeline.fetch")

PROGRAM = "fetch"
CATALOG_REFRESH_SEC = 3600.0   # --daemon: camera catalog refresh interval


class Fetcher:
    """Downloads the clips of `cameras` (None = every camera on the account) through one provider session."""

    def __init__(self, settings: Settings, provider: CameraProvider, cameras: Optional[list[str]],
                 filter_override: Optional[str] = None):
        self.settings = settings
        self.provider = provider
        self.cameras = cameras
        self.filter_override = filter_override  # --filter: one filter for every camera
        self.tz = ZoneInfo(settings.timezone)
        self.raw = make_storage(settings, RAW_VIDEOS)
        self.processed = make_storage(settings, PROCESSED_VIDEOS)
        self.ignored = make_storage(settings, IGNORED_VIDEOS)

    @property
    def label(self) -> str:
        return "all cameras" if self.cameras is None else ", ".join(sorted({c.strip() for c in self.cameras}))

    def filter_of(self, camera: str) -> str:
        """The camera's detection filter (CAMERA_FILTERS), unless --filter overrides it."""
        return self.filter_override or self.settings.camera_filter(camera)

    async def day_clips(self, day: date) -> list[ProviderClip]:
        """The day's clips (local time) of `cameras` - enabled or not, so that a
        clip of a camera not enabled is recorded as skipped - oldest first."""
        start = datetime.combine(day, datetime.min.time(), tzinfo=self.tz)
        clips = [c async for c in self.provider.list_clips(self.cameras, since=start, until=start + timedelta(days=1))]
        return sorted(clips, key=lambda c: c.recorded_at)

    def _key(self, clip: ProviderClip) -> str:
        """The clip's file: named by recorded_at (the start of the recording) —
        or, for a clip stored before recorded_at moved to the start, by the
        provider's own time, so it is not downloaded again."""
        if clip.provider_time is not None and clip.provider_time != clip.recorded_at:
            legacy = raw_clip_key(self.settings, clip.camera, clip.provider_time)
            if self._stored_as(legacy, clip):
                return legacy
        for shift in range(5):  # the name already taken by another clip (an older one, named by its end): next second
            key = raw_clip_key(self.settings, clip.camera, clip.recorded_at + timedelta(seconds=shift))
            if not self.raw.exists(metadata_key(key)) or self._stored_as(key, clip):
                return key
        return key

    def _stored_as(self, key: str, clip: ProviderClip) -> bool:
        """`key` holds this clip (its .metadata.json names the clip's provider id)."""
        md_key = metadata_key(key)
        if not self.raw.exists(md_key):
            return False
        try:
            md = self.raw.read_json(md_key)
        except ValueError:
            return False
        return str((md.get("original") or {}).get("provider_clip_id")) == clip.provider_clip_id

    def _write_metadata(self, clip: ProviderClip, key: str, download: Optional[dict] = None) -> dict:
        """The clip's .metadata.json; `download`: why the video is not downloaded
        (filtered / skipped; video_file names the file it would have had)."""
        md = clip_metadata(clip, self.tz, self.raw.folder, key)
        if download:
            md["download"] = download
        self.raw.write_json(metadata_key(key), md)
        return md

    def _decided_before(self, key: str) -> Optional[str]:
        """filtered / skipped if an earlier pass decided not to download the clip; else None."""
        md_key = metadata_key(key)
        if self.raw.exists(key) or not self.raw.exists(md_key):
            return None
        try:
            md = self.raw.read_json(md_key)
        except ValueError:
            return None
        if decided(md) == "filtered" and self.filter_override and filtered_by(md) != self.filter_override:
            return None  # --filter: re-checked on purpose
        return decided(md)

    def _filtered_before(self, key: str) -> Optional[str]:
        return "filtered" if self._decided_before(key) == "filtered" else None

    def _queue_for_processing(self, clip: ProviderClip, key: str) -> str:
        """Copy the clip's .metadata.json into the processed area's inbox; returns the inbox key."""
        src = metadata_key(key)
        dst = inbox_key(clip.camera, PurePosixPath(src).name)
        self.processed.write_bytes(dst, self.raw.read_bytes(src))
        return dst

    def print_list(self, clips: list[ProviderClip]) -> None:
        for clip in clips:
            key = self._key(clip)
            state = "stored" if self.raw.exists(key) else "filtered" if self._filtered_before(key) else ""
            labels = ",".join(clip.detections) or "-"
            print(f"{clip.recorded_at.astimezone(self.tz):%H:%M:%S}  {clip.camera.strip():<14} {labels:<16} {state}")

    def rewrite_metadata(self, clips: list[ProviderClip]) -> None:
        """Regenerate .metadata.json from the provider's current listing, only for clips we have."""
        rewritten = 0
        for clip in clips:
            key = self._key(clip)
            if self.raw.exists(key):
                self._write_metadata(clip, key)
                rewritten += 1
        log.info("done: rewrote %d metadata file(s); %d listed clip(s) not stored, skipped",
                 rewritten, len(clips) - rewritten)

    async def download(self, clips: list[ProviderClip], limit: Optional[int] = None, quiet: bool = False) -> int:
        """Download the clips not stored yet; returns the number downloaded. `quiet`
        (daemon): the summary is logged only when something was downloaded."""
        downloaded = skipped = filtered = 0
        for clip in clips:
            key = self._key(clip)
            labels = ",".join(clip.detections) or "-"
            if self.raw.exists(key):
                if not self.raw.exists(metadata_key(key)):
                    self._write_metadata(clip, key)
                skipped += 1
                continue
            if self._decided_before(key):
                filtered += 1  # decided on an earlier pass: final, whatever the settings say now
                continue
            when = f"{clip.recorded_at.astimezone(self.tz):%H:%M:%S}"
            if not self.settings.camera_enabled(clip.camera):
                self._write_metadata(clip, key, skipped_download(self.tz))
                filtered += 1
                log.debug("not downloaded (camera not enabled): %s %s", clip.camera.strip(), when)
                continue
            camera_filter = self.filter_of(clip.camera)
            if not passes(camera_filter, clip.detections):
                # Metadata only, marked as filtered, so later passes don't try again; and an ignored entry.
                download = filtered_download(camera_filter, clip.detections, self.tz)
                md = self._write_metadata(clip, key, download)
                self.ignored.write_json(metadata_key(key), {**md, "ignored": {
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "by": "filter",
                    "reason": download["reason"]}})
                filtered += 1
                log.info("not downloaded (filter %s): %s %s [%s] - metadata marked filtered: %s",
                         camera_filter, clip.camera.strip(), when, labels, self.raw.uri(metadata_key(key)))
                continue
            if limit is not None and downloaded >= limit:
                break
            size = await self.raw.write_stream(key, self.provider.download_clip(clip))
            self._write_metadata(clip, key)
            inbox = self._queue_for_processing(clip, key)
            downloaded += 1
            log.info("downloaded %s [%s] (%d KB) -> %s", self.raw.uri(key), labels, size // 1024,
                     self.processed.uri(inbox))
        if downloaded or not quiet:
            log.info("done: %d downloaded, %d already stored, %d not downloaded (filter / camera not enabled)",
                     downloaded, skipped, filtered)
        return downloaded

    async def daemon(self, interval: int, limit: Optional[int]) -> None:
        """Poll today's clips forever (the day follows midnight); a failed poll is retried on the next one."""
        log.info("daemon mode: polling %s every %ds for today's clips of %s; stored clips are skipped "
                 "(Ctrl+C stops)", self.provider.name, interval, self.label)
        idle, catalog_at = False, time.monotonic()
        while True:
            if time.monotonic() - catalog_at >= CATALOG_REFRESH_SEC:
                catalog_at = time.monotonic()
                await refresh_catalog(self.settings, self.provider)
            try:
                today = datetime.now(self.tz).date()
                downloaded = await self.download(await self.day_clips(today), limit, quiet=True)
            except Exception:  # noqa: BLE001 - a failed poll is retried on the next one
                log.exception("poll failed - retrying in %ds", interval)
                downloaded = 0
            if downloaded:
                idle = False
            elif not idle:
                log.info("waiting for new clips...")
                idle = True
            await asyncio.sleep(interval)


async def refresh_catalog(settings: Settings, provider: CameraProvider) -> None:
    try:
        await catalog.refresh(settings, provider)
    except Exception as e:  # noqa: BLE001 - the catalog is informational; downloading goes on
        log.warning("camera catalog refresh failed: %s: %s", type(e).__name__, e)


def selected_cameras(settings: Settings, requested: Optional[list[str]]) -> Optional[list[str]]:
    """The provider camera names to list: --camera (if given) narrows CAMERAS; CAMERAS=all
    with no --camera = every camera on the account (None). Raises ValueError for
    --camera names not enabled."""
    rejected = [c for c in requested or [] if not settings.camera_enabled(c)]
    if rejected:
        raise ValueError(f"camera(s) not enabled in CAMERAS: {rejected} (CAMERAS: {settings.cameras})")
    return requested or None  # None: every camera; those not enabled get a "skipped" record


async def run(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        cameras = selected_cameras(settings, args.camera)
    except ValueError as e:
        log.error("%s", e)
        return 1
    log.info("Filters: %s", f"{args.filter} for every camera (--filter)" if args.filter else summary(settings.camera_filters))

    try:
        async with make_provider(settings) as provider:
            await refresh_catalog(settings, provider)
            fetcher = Fetcher(settings, provider, cameras, args.filter)
            if args.daemon:
                await fetcher.daemon(args.interval or settings.poll_interval_seconds, args.limit)
                return 0
            day = args.date or datetime.now(fetcher.tz).date()
            clips = await fetcher.day_clips(day)
            log.info("%d clip(s) for %s on %s", len(clips), fetcher.label, day)
            if args.list:
                fetcher.print_list(clips)
            elif args.metadata_only:
                fetcher.rewrite_metadata(clips)
            else:
                await fetcher.download(clips, args.limit)
            return 0
    except ProviderError as e:
        log.error("%s", e)
        return 1
    except asyncio.CancelledError:
        log.info("daemon stopped")
        return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m blink_watcher fetch", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=date.fromisoformat, default=None,
                        help="Local date YYYY-MM-DD (default: today)")
    parser.add_argument("--camera", action="append", default=None,
                        help="Camera name, repeatable (default: every camera in CAMERAS)")
    parser.add_argument("--limit", type=int, default=None, help="Download at most N new clips this run")
    parser.add_argument("--list", action="store_true",
                        help="List the day's clips and their provider labels; download nothing")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Rewrite .metadata.json for clips already stored, from the provider's listing; "
                             "download no video")
    parser.add_argument("--daemon", action="store_true",
                        help="Keep running: poll the provider for today's clips and download the ones not yet "
                             "stored (--limit = per poll)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Seconds between polls with --daemon (default: POLL_INTERVAL_SECONDS)")
    parser.add_argument("--filter", choices=FILTER_NAMES, default=None,
                        help="One detection filter for every camera, instead of CAMERA_FILTERS")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Also show the other modules' internal INFO logs")
    args = parser.parse_args(argv)
    if args.daemon and (args.date or args.list or args.metadata_only):
        parser.error("--daemon always fetches today; it cannot be combined with --date, --list or --metadata-only")
    log_file = setup_logging(PROGRAM, args.verbose, log.name, "blink_watcher.providers")
    log.info("Downloader started (log file: %s)", log_file)

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        log.info("daemon stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
