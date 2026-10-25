"""--daemon mode of the inbox services (process, notify): process
the inbox, then keep polling it every POLL_INTERVAL seconds while it is
empty. An item that failed is held back before it is retried, RETRY_AFTER
seconds the first time and twice as long after each further failure (up to
RETRY_MAX), so a persistent failure does not loop; once it has failed
ALERT_AFTER_ATTEMPTS times in a row it is logged as an error (core.alerts
tells Telegram) and is still retried. Each batch is reported to `health`
(core.health): failing when every item of it failed and one of them was new,
so one bad item that keeps failing does not count but a broken API key or
service does. Ctrl+C stops.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, Mapping, Optional, TypeVar, Union

from blink_watcher.core.health import Health
from blink_watcher.settings import get_settings

_settings = get_settings()

POLL_INTERVAL = _settings.inbox_poll_sec                     # seconds between inbox checks while idle
RETRY_AFTER = _settings.inbox_retry_sec                      # seconds before a failed item is retried the first time
RETRY_MAX = _settings.inbox_retry_max_sec                    # longest wait between retries of an item
ALERT_AFTER_ATTEMPTS = _settings.inbox_alert_after_failures  # failures in a row that make it an error (Telegram)

T = TypeVar("T")


def run_daemon(log: logging.Logger, what: str,
               next_batch: Callable[[set[str]], list[T]],
               process: Callable[[list[T]], Union[Mapping[str, str], Iterable[str]]],
               health: Optional[Health] = None) -> int:
    """`next_batch(held)` -> the next items to process, without the keys in
    `held`; `process(batch)` -> the keys of the items that failed (a mapping:
    key -> the error)."""
    log.info(f"Daemon mode: polling the inbox every {POLL_INTERVAL:g}s; failed {what} are retried "
             f"after {RETRY_AFTER:g}s, then less often (Ctrl+C stops)")
    retry_at: dict[str, float] = {}
    attempts: dict[str, int] = {}  # failures in a row, per item
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
                if health:
                    health.beat()
                time.sleep(POLL_INTERVAL)
                continue
            idle = False
            result = process(batch)
            errors = dict(result) if isinstance(result, Mapping) else dict.fromkeys(result, "")
            new_failed = False
            for key, error in errors.items():
                n = attempts[key] = attempts.get(key, 0) + 1
                new_failed |= n == 1
                wait = min(RETRY_AFTER * 2 ** (n - 1), RETRY_MAX)
                retry_at[key] = time.monotonic() + wait
                if n == ALERT_AFTER_ATTEMPTS:
                    log.error(f"{key} failed {n} times in a row{f' ({error})' if error else ''}; "
                              f"still retried, every {RETRY_MAX / 60:g} min at most")
                else:
                    log.warning(f"{key} will be retried in {wait / 60:g} min (failure {n})")
            for key in _keys(batch, errors):
                attempts.pop(key, None)
            if health:
                if errors and len(errors) == len(batch):
                    if new_failed:
                        health.failed(f"all {len(batch)} {what} of the last run failed: "
                                      f"{next(iter(errors.values())) or 'see the log'}")
                    else:
                        health.beat()  # only items that failed before: their own problem
                else:
                    health.ok()
    except KeyboardInterrupt:
        log.info("Daemon stopped")
        return 0


def _keys(batch: list, errors: dict) -> list[str]:
    """Keys of the batch's items that went through (their failure count is reset)."""
    keys = [item[0] if isinstance(item, tuple) else getattr(item, "inbox_key", None) for item in batch]
    return [k for k in keys if k and k not in errors]


UNREADABLE_ALERT_SEC = _settings.inbox_unreadable_alert_sec  # an inbox file still unreadable this long is an error (Telegram)
_unreadable: dict[str, tuple[float, bool]] = {}  # key -> (first seen, error logged)


def unreadable(log: logging.Logger, key: str, error: Exception) -> None:
    """An inbox file that cannot be read is skipped (it may still be being
    written): a warning the first time, an error once it stays unreadable for
    UNREADABLE_ALERT_SEC, nothing in between (the inbox is read every second)."""
    now = time.monotonic()
    first, logged = _unreadable.get(key, (now, False))
    if key not in _unreadable:
        log.warning(f"Unreadable inbox file {key}: {error} - skipped, left in place")
    elif not logged and now - first >= UNREADABLE_ALERT_SEC:
        log.error(f"Inbox file {key} is still unreadable ({error}) - fix or delete it")
        logged = True
    _unreadable[key] = (first, logged)
