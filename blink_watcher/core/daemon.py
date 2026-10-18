"""--daemon mode of the inbox services (process, notify): process
the inbox, then keep polling it every POLL_INTERVAL seconds while it is
empty. An item that failed is held back for RETRY_AFTER seconds before it is
retried, so a persistent failure does not loop every second. Ctrl+C stops.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, TypeVar

POLL_INTERVAL = 1.0   # seconds between inbox checks while idle
RETRY_AFTER = 60.0    # seconds before a failed item is retried

T = TypeVar("T")


def run_daemon(log: logging.Logger, what: str,
               next_batch: Callable[[set[str]], list[T]],
               process: Callable[[list[T]], Iterable[str]]) -> int:
    """`next_batch(held)` -> the next items to process, without the keys in
    `held`; `process(batch)` -> the keys of the items that failed."""
    log.info(f"Daemon mode: polling the inbox every {POLL_INTERVAL:g}s; failed {what} are retried "
             f"after {RETRY_AFTER:g}s (Ctrl+C stops)")
    retry_at: dict[str, float] = {}
    idle = False
    try:
        while True:
            now = time.monotonic()
            retry_at = {k: t for k, t in retry_at.items() if t > now}
            batch = next_batch(set(retry_at))
            if not batch:
                if not idle:
                    held = f" ({len(retry_at)} failed, waiting for retry)" if retry_at else ""
                    log.info(f"Waiting for new {what}{held}...")
                    idle = True
                time.sleep(POLL_INTERVAL)
                continue
            idle = False
            for key in process(batch):
                retry_at[key] = time.monotonic() + RETRY_AFTER
                log.warning(f"{key} will be retried in {RETRY_AFTER:g}s")
    except KeyboardInterrupt:
        log.info("Daemon stopped")
        return 0
