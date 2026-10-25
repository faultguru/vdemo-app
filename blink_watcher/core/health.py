"""Service health: so that a service failing over and over is noticed, not just logged.

Each service (fetch, process, notify, chat-bot) keeps a Health and reports
every loop pass as ok() or failed(error). The outside services they depend
on are reported the same way by whoever uses them (a Dependency): blink by
fetch's polls, telegram by the chat-bot, ai by every LLM call (any program;
the file is shared, the last call wins). Each state is written to
STORAGE_ROOT/state/health/<name>.json (only when it changes, and at most
every HEARTBEAT_SEC otherwise, as a heartbeat):

    {"program", "label", "type": "service" | "dependency", "updated_at", "ok",
     "last_ok_at", "failures", "failing_since", "error", "alerted"}

Once a service has been failing for HEALTH_ALERT_AFTER_MINUTES a message is
queued for Telegram (chat/outbox/<stamp>-<PROGRAM>_FAILING.message.json), and
another when it works again (<stamp>-<PROGRAM>_RECOVERED). The supervisor
reads the same files to spot a service that stopped reporting (hung) and
crash loops (supervisor.Watchdog); the web UI shows a banner from them.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from blink_watcher.core.runtime import state_dir

log = logging.getLogger("blink_watcher.core.health")

HEARTBEAT_SEC = 60.0  # an unchanged healthy state is rewritten this often
LABELS = {             # what stops working when the service fails, for the messages
    "fetch": "camera downloads",
    "process": "clip processing",
    "notify": "notifications",
    "chat-bot": "Telegram bot",
    "blink": "Blink cloud",
    "telegram": "Telegram",
}
SERVICES = ("fetch", "process", "notify", "chat-bot")  # in the order of the status page
DEPENDENCIES = ("blink", "telegram", "ai")


def health_dir(settings) -> Path:
    return state_dir(settings, "health")


def read_all(settings) -> dict[str, dict]:
    """program -> its last reported state."""
    out = {}
    for p in sorted(health_dir(settings).glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return out


def tell(settings, name: str, text: str) -> bool:
    """Queue a system message for Telegram (chat/outbox/<stamp>-<name>); True once queued."""
    from blink_watcher.chat import store as chat_store
    now = datetime.now(ZoneInfo(settings.timezone))
    try:
        chat_store.queue_message(settings.chat_dir, f"{now:%Y-%m-%d-%H-%M-%S}-{name}",
                                 {"date": f"{now:%Y-%m-%d}", "text": text})
    except OSError as e:
        log.error("could not queue the message for Telegram: %s", e)
        return False
    log.info("Telegram message queued: %s", text)
    return True


class Health:
    type = "service"

    def __init__(self, settings, program: str, label: Optional[str] = None,
                 alert: bool = True, log_name: Optional[str] = None):
        """alert: tell Telegram when it keeps failing (and when it is fixed);
        log_name: the log those messages point to (default: program)."""
        self.settings = settings
        self.program = program
        self.label = label or LABELS.get(program, program)
        self.alert = alert
        self.log_name = log_name or program
        self.tz = ZoneInfo(settings.timezone)
        self.path = health_dir(settings) / f"{program}.json"
        self.failures = 0
        self.failing_since: Optional[datetime] = None
        self.error: Optional[str] = None
        self.alerted = False
        self.last_ok_at: Optional[datetime] = None
        self._written = 0.0

    def ok(self) -> None:
        recovered = self.failures > 0
        if recovered:
            log.info("%s working again after %d failure(s) since %s", self.label, self.failures,
                     f"{self.failing_since:%H:%M}")
            if self.alerted and self.alert:
                tell(self.settings, f"{self.program.upper()}_RECOVERED",
                     f"Fixed: {self.label} working again (failing since {self.failing_since:%H:%M}).")
        self.failures, self.failing_since, self.error, self.alerted = 0, None, None, False
        self.last_ok_at = datetime.now(self.tz)
        self._write(force=recovered)

    def failed(self, error: str) -> None:
        self.failures += 1
        self.failing_since = self.failing_since or datetime.now(self.tz)
        self.error = error
        after = self.settings.health_alert_after_minutes
        if (self.alert and not self.alerted and after > 0
                and datetime.now(self.tz) - self.failing_since >= timedelta(minutes=after)):
            self.alerted = tell(self.settings, f"{self.program.upper()}_FAILING",
                                f"Problem: {self.label} failing since {self.failing_since:%H:%M} "
                                f"({error}). Still retrying; see logs/{self.log_name}.log.")
        self._write(force=True)

    def beat(self) -> None:
        """Still alive (e.g. idle): rewrite the state if it is getting old."""
        self._write()

    def _write(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._written < HEARTBEAT_SEC:
            return
        iso = lambda t: t.isoformat(timespec="seconds") if t else None  # noqa: E731
        state = {"program": self.program, "label": self.label, "type": self.type,
                 "updated_at": iso(datetime.now(self.tz)),
                 "ok": self.failures == 0, "last_ok_at": iso(self.last_ok_at), "failures": self.failures,
                 "failing_since": iso(self.failing_since), "error": self.error, "alerted": self.alerted}
        for attempt in range(3):  # Windows: the replace fails while a reader has the file open
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".json.part")
                tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, self.path)
                self._written = time.monotonic()
                return
            except OSError as e:
                if attempt == 2:
                    log.warning("could not write %s: %s", self.path, e)
                else:
                    time.sleep(0.2)


class Dependency(Health):
    """An outside service (blink, telegram, ai), as seen by the programs that use it.
    Several processes may report the same one (ai): each report first takes over
    the state in the file, so the last call decides and a failure seen by one
    process is cleared by the next success seen by any."""
    type = "dependency"

    def __init__(self, settings, name: str, label: Optional[str] = None,
                 alert: bool = False, log_name: Optional[str] = None):
        super().__init__(settings, name, label, alert, log_name)

    def _sync(self) -> None:
        try:
            h = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        when = lambda v: datetime.fromisoformat(v) if v else None  # noqa: E731
        try:
            self.failures = int(h.get("failures") or 0)
            self.failing_since, self.last_ok_at = when(h.get("failing_since")), when(h.get("last_ok_at"))
        except (TypeError, ValueError):
            return
        self.error, self.alerted = h.get("error"), bool(h.get("alerted"))
        if self.failures and self.failing_since is None:
            self.failing_since = datetime.now(self.tz)

    def ok(self) -> None:
        self._sync()
        super().ok()

    def failed(self, error: str) -> None:
        self._sync()
        super().failed(error)
