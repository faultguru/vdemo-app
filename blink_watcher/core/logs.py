"""Logging for the programs and services: stdlib logging via dictConfig, to
stdout and to a rotating file per program, STORAGE_ROOT/logs/<program>.log.
Each line names the program that wrote it, e.g. "[fetch]", so lines from
several services can be told apart when read together; while a clip or an
event is being handled (log_clip), also its camera and recording time, e.g.
"[process] [Front Door 2026-10-24 10:59:55]". Errors of the services started by
`run` also go to Telegram (core.alerts)."""
from __future__ import annotations

import logging.config
import os
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(program)s] %(clip)s%(message)s"
_clip: ContextVar[str] = ContextVar("log_clip", default="")


def log_clip(camera: Optional[str], recorded_at: Union[str, datetime, None] = None) -> None:
    """Every log line from here on (in this thread / task) starts with the clip's
    camera and local recording time; log_clip(None) ends it."""
    if not camera:
        _clip.set("")
        return
    when = (f"{recorded_at:%Y-%m-%d %H:%M:%S}" if isinstance(recorded_at, datetime)
            else (recorded_at or "")[:19].replace("T", " "))
    _clip.set(f"[{camera.strip()}{' ' + when if when else ''}] ")
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ProgramFilter(logging.Filter):
    """Adds `program`: the name of the running program (daemon); and `clip`: the
    clip or event being handled (log_clip), or nothing."""

    def __init__(self, program: str):
        super().__init__()
        self.program = program

    def filter(self, record: logging.LogRecord) -> bool:
        record.program = self.program
        record.clip = _clip.get()
        return True


class ConnectionResetFilter(logging.Filter):
    """Drops asyncio's "Exception in callback ..._call_connection_lost" tracebacks:
    on Windows every connection the other side closes (a browser reload, a
    dropped HTTP connection) logs one as an ERROR, and it is harmless."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.exc_info and isinstance(record.exc_info[1], ConnectionResetError))


SUPERVISED = "BLINK_WATCHER_SUPERVISED"  # set by `run` for the services it starts


def setup_logging(program: str, verbose: bool, *loggers: str) -> Path:
    """stdout + log_file (rotated at 10 MB, 5 kept). `loggers` log at INFO;
    every other module only shows warnings unless verbose. Under `run` (and for
    `run` itself), errors are also sent to Telegram (core.alerts, ALERT_ON_ERRORS)."""
    from blink_watcher.core.runtime import logs_dir
    from blink_watcher.settings import get_settings

    settings = get_settings()
    log_file = logs_dir(settings) / f"{program}.log"
    sys.stdout.reconfigure(errors="replace")  # Windows consoles can't print every character Claude returns
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": LOG_FORMAT, "datefmt": LOG_DATE_FORMAT}},
        "filters": {"program": {"()": ProgramFilter, "program": program},
                    "resets": {"()": ConnectionResetFilter}},
        "handlers": {
            "stdout": {"class": "logging.StreamHandler", "stream": "ext://sys.stdout", "formatter": "default",
                       "filters": ["program"]},
            "file": {"class": "logging.handlers.RotatingFileHandler", "filename": str(log_file),
                     "maxBytes": 10 * 2**20, "backupCount": 5, "encoding": "utf-8", "formatter": "default",
                     "filters": ["program"]},
        },
        "root": {"level": "INFO" if verbose else "WARNING", "handlers": ["stdout", "file"]},
        "loggers": {"asyncio": {"filters": ["resets"]}, **{name: {"level": "INFO"} for name in loggers}},
    })
    if settings.alert_on_errors and (program == "run" or os.environ.get(SUPERVISED) == "1"):
        from blink_watcher.core.alerts import TelegramAlerts
        alerts = TelegramAlerts(settings, program)
        alerts.addFilter(ProgramFilter(program))
        logging.getLogger().addHandler(alerts)
    return log_file
