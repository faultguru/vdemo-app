import logging
from datetime import datetime, timedelta, timezone

from blinkpy.blinkpy import Blink

from .blink_api import list_clips
from .config import Settings
from .store import Store

log = logging.getLogger(__name__)

MAX_HOURS = 60 * 24
CHUNK_HOURS = 6


async def run_backfill(
    settings: Settings,
    store: Store,
    blink: Blink,
    hours: int,
    restart: bool = False,
    dry_run: bool = False,
) -> int:
    hours = max(1, min(hours, MAX_HOURS))
    now = datetime.now(timezone.utc)
    target = now - timedelta(hours=hours)

    state = store.load_backfill()
    if state and not restart:
        cursor = datetime.fromisoformat(state["cursor_ts"])
        prior_target = datetime.fromisoformat(state["target_ts"])
        target = min(target, prior_target)
        log.info("resuming backfill from cursor=%s → target=%s", cursor, target)
    else:
        cursor = now
        log.info("starting backfill: now=%s → target=%s (%d hours)", cursor, target, hours)
        if restart:
            store.clear_backfill()

    total_enqueued = 0
    while cursor > target:
        window_start = max(cursor - timedelta(hours=CHUNK_HOURS), target)
        window_enqueued = 0

        async for clip in list_clips(
            blink, settings.cameras, since=window_start, until=cursor
        ):
            if dry_run:
                window_enqueued += 1
                continue
            with store.tx():
                if store.enqueue(clip.id, clip.camera, clip.ts, clip.source_url, priority=1):
                    window_enqueued += 1

        log.info(
            "backfill: %s → %s  +%d clips  (cursor -%s)",
            window_start.isoformat(timespec="minutes"),
            cursor.isoformat(timespec="minutes"),
            window_enqueued,
            now - window_start,
        )
        total_enqueued += window_enqueued
        cursor = window_start
        if not dry_run:
            with store.tx():
                store.save_backfill(cursor, target)

    if not dry_run:
        with store.tx():
            store.clear_backfill()
    log.info("backfill complete: %d clips enqueued (dry_run=%s)", total_enqueued, dry_run)
    return total_enqueued
