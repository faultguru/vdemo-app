"""Errors to Telegram: every ERROR (or worse) logged by this project's code is
also queued for Telegram (core.health.tell), so a problem that needs the user
is not only in a log file. ERROR is kept for that: a failure that is retried
on its own is logged as a WARNING (core.health tells Telegram if it goes on).

Each kind of error (its message with the numbers left out) is sent once per
ALERT_REPEAT_MINUTES; the repeats meanwhile are counted and mentioned in the
next message. At most MAX_PER_HOUR messages an hour per program. What was sent
is kept in STORAGE_ROOT/state/alerts/<program>.json, so a restarted service
does not send the same message again. A log call can opt out with
extra={"alert": False} (when it has told Telegram itself).

Installed by core.logs.setup_logging for the services started by `run`
(ALERT_ON_ERRORS).
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime

from blink_watcher.core.runtime import state_dir
from blink_watcher.settings import get_settings

_settings = get_settings()

MAX_PER_HOUR = _settings.alert_max_per_hour
KEEP_SEC = 2 * 24 * 3600          # forget a kind of error not seen for this long
OWN_LOGGERS = ("blink_watcher.",)  # third-party libraries log their own view of the same failures
SKIP_LOGGERS = ("blink_watcher.core.health", __name__)  # they queue the messages: no loops


def _kind(text: str) -> str:
    return re.sub(r"\d+", "#", text)[:160]


class TelegramAlerts(logging.Handler):
    def __init__(self, settings, program: str):
        super().__init__(logging.ERROR)
        self.settings = settings
        self.program = program
        self.repeat_sec = settings.alert_repeat_minutes * 60
        self.path = state_dir(settings, "alerts") / f"{program}.json"
        try:
            self.sent: dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.sent = {}

    def emit(self, record: logging.LogRecord) -> None:
        if (not record.name.startswith(OWN_LOGGERS) or record.name.startswith(SKIP_LOGGERS)
                or getattr(record, "alert", True) is False):
            return
        try:
            self._alert(record)
        except Exception:  # noqa: BLE001 - an alert must never break the logging call
            pass

    def _alert(self, record: logging.LogRecord) -> None:
        message = record.getMessage().strip()
        key = _kind(message)
        now = time.time()
        self.sent = {k: v for k, v in self.sent.items() if now - v["at"] < KEEP_SEC}
        prev = self.sent.get(key)
        if prev and now - prev["at"] < self.repeat_sec:
            prev["repeats"] = prev.get("repeats", 0) + 1
            self._save()
            return
        if sum(1 for v in self.sent.values() if now - v["at"] < 3600) >= MAX_PER_HOUR:
            return
        error = record.exc_info[1] if record.exc_info else None
        detail = f" ({type(error).__name__}: {error})" if error and str(error) not in message else ""
        repeats = (f" Happened {prev['repeats']} more time(s) since "
                   f"{datetime.fromtimestamp(prev['at']):%m-%d %H:%M}." if prev and prev.get("repeats") else "")
        clip = getattr(record, "clip", "")
        text = f"Problem in {self.program}: {clip}{message}{detail}".rstrip(". ")
        text = (text[:700] + "...") if len(text) > 700 else text
        from blink_watcher.core.health import tell
        if tell(self.settings, f"{self.program.upper()}_ERROR-{int(now * 1000) % 1000:03d}",
                f"{text}.{repeats} See logs/{self.program}.log."):
            self.sent[key] = {"at": now, "repeats": 0}
            self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.sent, indent=1, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
