"""Crash-safe tracing for hardware-accelerated steps (GPU inference, GPU encode).

The GPU work can hard-hang the machine (no blue screen, no dump), so normal
logging - buffered, and lost with the OS file cache on a hard reset - can't say
where it died. trace() appends one JSON line per event to
STORAGE_ROOT/logs/hw_trace.log and fsyncs it before returning, so after a
reboot the last line is the last step that started or finished.

Pair it with `python -m blink_watcher hwmonitor`, which records GPU/CPU/memory
load every few seconds the same way.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from blink_watcher.core.runtime import logs_dir
from blink_watcher.settings import get_settings

TRACE_FILE = "hw_trace.log"
_lock = threading.Lock()


def durable_append(path: Path, line: str) -> None:
    """Append a line and force it to disk (survives a hard reset)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())


def trace(event: str, **fields) -> None:
    rec = {"ts": datetime.now().isoformat(timespec="milliseconds"), "pid": os.getpid(), "event": event, **fields}
    durable_append(logs_dir(get_settings()) / TRACE_FILE, json.dumps(rec, default=str))


@contextmanager
def traced(event: str, **fields):
    """trace <event>.start before, <event>.end (with seconds) or <event>.error after."""
    trace(f"{event}.start", **fields)
    t0 = time.perf_counter()
    try:
        yield
    except BaseException as e:
        trace(f"{event}.error", sec=round(time.perf_counter() - t0, 3), error=f"{type(e).__name__}: {e}"[:500], **fields)
        raise
    trace(f"{event}.end", sec=round(time.perf_counter() - t0, 3), **fields)
