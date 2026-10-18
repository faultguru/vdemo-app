"""Logging for the programs and services: stdlib logging via dictConfig, to
stdout and to a rotating file per program, STORAGE_ROOT/logs/<program>.log.
Each line names the program that wrote it, e.g. "[fetch]", so lines from
several services can be told apart when read together."""
from __future__ import annotations

import logging.config
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(program)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ProgramFilter(logging.Filter):
    """Adds `program`: the name of the running program (daemon)."""

    def __init__(self, program: str):
        super().__init__()
        self.program = program

    def filter(self, record: logging.LogRecord) -> bool:
        record.program = self.program
        return True


def setup_logging(program: str, verbose: bool, *loggers: str) -> Path:
    """stdout + log_file (rotated at 10 MB, 5 kept). `loggers` log at INFO;
    every other module only shows warnings unless verbose."""
    from blink_watcher.core.runtime import logs_dir
    from blink_watcher.settings import get_settings

    log_file = logs_dir(get_settings()) / f"{program}.log"
    sys.stdout.reconfigure(errors="replace")  # Windows consoles can't print every character Claude returns
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": LOG_FORMAT, "datefmt": LOG_DATE_FORMAT}},
        "filters": {"program": {"()": ProgramFilter, "program": program}},
        "handlers": {
            "stdout": {"class": "logging.StreamHandler", "stream": "ext://sys.stdout", "formatter": "default",
                       "filters": ["program"]},
            "file": {"class": "logging.handlers.RotatingFileHandler", "filename": str(log_file),
                     "maxBytes": 10 * 2**20, "backupCount": 5, "encoding": "utf-8", "formatter": "default",
                     "filters": ["program"]},
        },
        "root": {"level": "INFO" if verbose else "WARNING", "handlers": ["stdout", "file"]},
        "loggers": {name: {"level": "INFO"} for name in loggers},
    })
    return log_file
