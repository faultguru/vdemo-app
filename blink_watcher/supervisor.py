"""Every service in one console: starts fetch, process, notify, chat-bot and
webui as child processes, shows their output together and keeps them running.

    python -m blink_watcher run                       # every service
    python -m blink_watcher run --only webui          # just these (comma-separated or repeated)
    python -m blink_watcher run --skip chat-bot,webui # all but these

Each service stays its own Python process (YOLO is CPU-heavy, the loops
block), started as `python -m blink_watcher <command>`. Their log lines already
name the program ("[fetch]"); any other output line (a traceback, a print) is
prefixed with it. A child that exits is restarted after RESTART_MIN seconds,
doubling up to RESTART_MAX while it keeps failing; one that ran STABLE_AFTER
seconds starts the backoff over. Ctrl+C stops every child the way Ctrl+C stops
it when run on its own (Windows: CTRL_BREAK to its process group, which cli
turns into KeyboardInterrupt; POSIX: SIGINT), then waits up to STOP_TIMEOUT
seconds before killing it.

The supervisor logs starts, exits and restarts to STORAGE_ROOT/logs/run.log;
each service keeps writing its own log file there too.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import IO, Optional

from blink_watcher.core.logs import setup_logging
from blink_watcher.core.paths import PROJECT_ROOT

log = logging.getLogger("blink_watcher.supervisor")

PROGRAM = "run"
RESTART_MIN = 2.0      # seconds before the first restart of a child that exited
RESTART_MAX = 60.0     # longest wait between restarts
STABLE_AFTER = 60.0    # a child that ran this long restarts after RESTART_MIN again
STOP_TIMEOUT = 20.0    # seconds a child gets to stop on Ctrl+C before it is killed
LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")
WINDOWS = os.name == "nt"


@dataclass
class Service:
    name: str
    args: list[str]                          # the child's command line, without the interpreter
    proc: Optional[subprocess.Popen] = None
    started_at: float = 0.0
    restart_at: Optional[float] = None       # monotonic time of the next start, while it is down
    backoff: float = RESTART_MIN
    reader: Optional[threading.Thread] = field(default=None, repr=False)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def default_services() -> list[Service]:
    cli = [sys.executable, "-m", "blink_watcher"]
    return [
        Service("fetch", cli + ["fetch", "--daemon"]),
        Service("process", cli + ["process", "--daemon"]),
        Service("notify", cli + ["notify", "--daemon"]),
        Service("chat-bot", cli + ["chat-bot"]),
        Service("webui", cli + ["webui"]),
    ]


class Supervisor:
    def __init__(self, services: list[Service], out: IO[str] = sys.stdout):
        self.services = services
        self.out = out
        self.out_lock = threading.Lock()
        self.stopping = False

    # ---------- children ----------

    def start(self, svc: Service) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        # Own process group / session: a console Ctrl+C reaches only the
        # supervisor, which then stops each child in turn.
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}
        svc.proc = subprocess.Popen(svc.args, cwd=PROJECT_ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding="utf-8", errors="replace", bufsize=1, **flags)
        svc.started_at, svc.restart_at = time.monotonic(), None
        svc.reader = threading.Thread(target=self._pump, args=(svc, svc.proc.stdout), daemon=True,
                                      name=f"output-{svc.name}")
        svc.reader.start()
        log.info(f"{svc.name} started (pid {svc.proc.pid})")

    def _pump(self, svc: Service, stream: IO[str]) -> None:
        """Copies a child's output to the console, line by line."""
        for line in stream:
            line = line.rstrip("\r\n")
            if not LOG_LINE.match(line):
                line = f"[{svc.name}] {line}"
            with self.out_lock:
                self.out.write(line + "\n")
                self.out.flush()

    def _exited(self, svc: Service, now: float) -> None:
        code = svc.proc.returncode
        ran = now - svc.started_at
        if ran >= STABLE_AFTER:
            svc.backoff = RESTART_MIN
        delay = svc.backoff
        svc.backoff = min(svc.backoff * 2, RESTART_MAX)
        svc.restart_at = now + delay
        log.log(logging.ERROR if code else logging.WARNING,
                f"{svc.name} exited with code {code} after {ran:.0f}s; restarting in {delay:g}s")
        svc.proc = None

    def _interrupt(self, svc: Service) -> None:
        try:
            if WINDOWS:
                svc.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                svc.proc.send_signal(signal.SIGINT)
        except OSError:
            svc.proc.terminate()

    @staticmethod
    def _kill(svc: Service) -> None:
        """The child and whatever it started (ffmpeg, an LLM CLI call)."""
        try:
            if WINDOWS:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(svc.proc.pid)], capture_output=True, check=False)
            else:
                os.killpg(svc.proc.pid, signal.SIGKILL)  # its own session: pgid == pid
        except OSError:
            pass
        svc.proc.kill()

    def stop_all(self) -> None:
        self.stopping = True
        live = [s for s in self.services if s.running]
        if not live:
            return
        log.info(f"Stopping {', '.join(s.name for s in live)}...")
        for svc in live:
            self._interrupt(svc)
        deadline = time.monotonic() + STOP_TIMEOUT
        for svc in live:
            try:
                svc.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
                log.info(f"{svc.name} stopped (code {svc.proc.returncode})")
            except subprocess.TimeoutExpired:
                log.warning(f"{svc.name} did not stop in {STOP_TIMEOUT:g}s; killed")
                self._kill(svc)
                svc.proc.wait()
        for svc in live:
            if svc.reader:
                svc.reader.join(timeout=2)

    # ---------- main loop ----------

    def run(self) -> int:
        log.info(f"Starting {', '.join(s.name for s in self.services)} (Ctrl+C stops)")
        for svc in self.services:
            self.start(svc)
        try:
            while True:
                time.sleep(0.5)
                now = time.monotonic()
                for svc in self.services:
                    if svc.proc is not None and svc.proc.poll() is not None:
                        if svc.reader:
                            svc.reader.join(timeout=2)  # its last lines first
                        self._exited(svc, now)
                    elif svc.proc is None and svc.restart_at is not None and now >= svc.restart_at:
                        self.start(svc)
        except KeyboardInterrupt:
            self.stop_all()
            log.info("All services stopped")
            return 0


def _names(values: Optional[list[str]]) -> list[str]:
    return [n.strip() for v in values or [] for n in v.split(",") if n.strip()]


def select(services: list[Service], only: list[str], skip: list[str]) -> list[Service]:
    known = {s.name for s in services}
    unknown = sorted(set(only + skip) - known)
    if unknown:
        raise SystemExit(f"unknown service(s): {', '.join(unknown)} (services: {', '.join(sorted(known))})")
    chosen = [s for s in services if (not only or s.name in only) and s.name not in skip]
    if not chosen:
        raise SystemExit("no service left to run")
    return chosen


def main(argv: Optional[list[str]] = None) -> int:
    services = default_services()
    names = ", ".join(s.name for s in services)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", metavar="NAMES", help=f"Run only these services ({names})")
    parser.add_argument("--skip", action="append", metavar="NAMES", help="Run every service but these")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also show other modules' INFO logs")
    args = parser.parse_args(argv)
    chosen = select(services, _names(args.only), _names(args.skip))
    setup_logging(PROGRAM, args.verbose, log.name)
    if not WINDOWS:  # a SIGTERM (service manager, kill) stops the children like Ctrl+C
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    return Supervisor(chosen).run()
