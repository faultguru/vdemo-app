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
seconds before killing it; a second Ctrl+C kills the ones still running.

Settings changes apply on their own: the supervisor watches .env and, once it
has been unchanged for SETTLE_SEC, restarts the services the changed options
belong to (the "@programs" of their group in .env.example; every service for
an option not documented there). A restarting service is stopped the Ctrl+C
way and started again at once; clips and events wait on disk, so a clip that
was being processed starts over.

A watchdog (every WATCHDOG_SEC) applies the security mode changes that are due
(scheduled or timed, core.security_mode), and catches what a service cannot report itself
(core.health): fetch is restarted when its health file has not been updated
for HEALTH_STALE_MINUTES (it is stuck), and Telegram is told; and when a
service exits CRASH_ALERT_AFTER_RESTARTS times in a row without staying up for
STABLE_AFTER seconds, Telegram is told too (and again once it stays up).
After the computer was asleep (the clock jumped more than ASLEEP_SEC between
two checks), the services get HEALTH_STALE_MINUTES again before they count as
stuck: they could not report while it slept.

The supervisor logs starts, exits and restarts to STORAGE_ROOT/logs/run.log;
each service keeps writing its own log file there too. Only one runs at a
time: a second one exits at once. It also stops when state/run.stop appears
(`vdemo stop`, blink_watcher.control); with --background (`vdemo start`, no
console to show it) the services' output that is not already a log line (a
traceback before logging is set up) goes to run.log instead.
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
from datetime import datetime
from typing import IO, Optional

from dotenv import dotenv_values

from blink_watcher.core.logs import SUPERVISED as logs_supervised, setup_logging
from blink_watcher.core.paths import ENV_FILE, PROJECT_ROOT


def _config():
    """The settings for the constants below; their defaults when .env cannot be
    loaded (a bad .env must not stop the supervisor)."""
    try:
        from blink_watcher.settings import get_settings
        return get_settings()
    except (Exception, SystemExit):  # noqa: BLE001
        from blink_watcher.settings import Settings
        return Settings.model_construct()


_settings = _config()

log = logging.getLogger("blink_watcher.supervisor")

PROGRAM = "run"
RESTART_MIN = _settings.restart_min_sec      # seconds before the first restart of a child that exited
RESTART_MAX = _settings.restart_max_sec      # longest wait between restarts
STABLE_AFTER = _settings.restart_stable_sec  # a child that ran this long restarts after RESTART_MIN again
STOP_TIMEOUT = _settings.stop_timeout_sec    # seconds a child gets to stop on Ctrl+C before it is killed
SETTLE_SEC = _settings.settings_settle_sec   # .env unchanged this long after a change: the affected services restart
WATCHDOG_SEC = _settings.watchdog_sec        # how often the watchdog looks at the services' health
ASLEEP_SEC = _settings.asleep_detect_sec     # a wall-clock jump this big between two loop passes: the computer slept
HEARTBEAT_SERVICES = ("fetch",)  # restarted when their health file goes stale (the others may
                                 # legitimately spend long on one batch)
SUPERVISED = logs_supervised  # set for the children: settings changes are applied for them, errors go to Telegram
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
    reload_by: Optional[float] = None        # stopping to apply settings: monotonic kill deadline
    started_wall: float = 0.0                # time.time() of the last start
    quick_exits: int = 0                     # exits in a row before running STABLE_AFTER seconds
    crash_alerted: bool = False
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
    def __init__(self, services: list[Service], out: IO[str] = sys.stdout, background: bool = False):
        self.services = services
        self.out = out
        self.background = background  # no console: only the lines no log file has go to run.log
        self.out_lock = threading.Lock()
        self.stopping = False

    # ---------- children ----------

    def start(self, svc: Service) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", SUPERVISED: "1"}
        # Own process group / session: a console Ctrl+C reaches only the
        # supervisor, which then stops each child in turn.
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}
        svc.proc = subprocess.Popen(svc.args, cwd=PROJECT_ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding="utf-8", errors="replace", bufsize=1, **flags)
        svc.started_at, svc.started_wall, svc.restart_at = time.monotonic(), time.time(), None
        svc.reader = threading.Thread(target=self._pump, args=(svc, svc.proc.stdout), daemon=True,
                                      name=f"output-{svc.name}")
        svc.reader.start()
        log.info(f"{svc.name} started (pid {svc.proc.pid})")

    def _pump(self, svc: Service, stream: IO[str]) -> None:
        """Copies a child's output to the console, line by line."""
        for line in stream:
            line = line.rstrip("\r\n")
            if self.background:
                if line and not LOG_LINE.match(line):
                    log.info(f"[{svc.name}] {line}")
                continue
            if not LOG_LINE.match(line):
                line = f"[{svc.name}] {line}"
            with self.out_lock:
                self.out.write(line + "\n")
                self.out.flush()

    def _exited(self, svc: Service, now: float) -> None:
        code = svc.proc.returncode
        ran = now - svc.started_at
        if svc.reload_by is not None:  # stopped to apply settings: back at once, no backoff
            svc.reload_by, svc.proc, svc.restart_at = None, None, now
            log.info(f"{svc.name} stopped (code {code}) - starting again")
            return
        if ran >= STABLE_AFTER:
            svc.backoff = RESTART_MIN
            svc.quick_exits = 0
        svc.quick_exits += 1
        delay = svc.backoff
        svc.backoff = min(svc.backoff * 2, RESTART_MAX)
        svc.restart_at = now + delay
        log.log(logging.ERROR if code else logging.WARNING,  # Telegram is told below if it keeps exiting
                f"{svc.name} exited with code {code} after {ran:.0f}s; restarting in {delay:g}s",
                extra={"alert": False})
        svc.proc = None
        limit = _setting(self.settings(), "crash_alert_after_restarts", 0)
        if limit > 0 and svc.quick_exits >= limit and not svc.crash_alerted:
            svc.crash_alerted = self._tell(f"{svc.name.upper()}_CRASHING",
                                           f"Problem: the {svc.name} service keeps stopping ({svc.quick_exits} "
                                           f"times in a row, last exit code {code}). Restarting it; see "
                                           f"logs/{svc.name}.log.")

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

    def kill_all(self) -> None:
        for svc in self.services:
            if svc.running:
                self._kill(svc)
                log.info(f"{svc.name} killed")

    # ---------- settings ----------

    def _env_state(self) -> tuple[Optional[float], dict]:
        try:
            return ENV_FILE.stat().st_mtime, dict(dotenv_values(ENV_FILE))
        except OSError:
            return None, {}

    def _check_settings(self, now: float) -> None:
        """A settled .env change: restart the services its options belong to."""
        try:
            mtime = ENV_FILE.stat().st_mtime
        except OSError:
            mtime = None
        if mtime != self.env_mtime:
            self.env_mtime, self.env_changed_at = mtime, now
            return
        if self.env_changed_at is None or now - self.env_changed_at < SETTLE_SEC:
            return
        self.env_changed_at = None
        _, values = self._env_state()
        changed = sorted(k for k in set(values) | set(self.env_values) if values.get(k) != self.env_values.get(k))
        self.env_values = values
        if not changed:
            return
        names = affected(changed)
        targets = [s for s in self.services if s.name in names]
        log.info(f"Settings changed ({', '.join(changed)}) - restarting "
                 f"{', '.join(s.name for s in targets) or 'nothing (no running service uses them)'}")
        for svc in targets:
            if svc.running:
                svc.reload_by = now + STOP_TIMEOUT
                self._interrupt(svc)
            else:
                svc.restart_at, svc.backoff = now, RESTART_MIN  # down (waiting to restart): start it now

    # ---------- watchdog ----------

    def settings(self):
        """The current settings (None if they cannot be loaded: the watchdog then does nothing)."""
        try:
            from blink_watcher.settings import load_settings
            return load_settings()
        except (Exception, SystemExit) as e:  # noqa: BLE001 - a bad .env must not stop the supervisor
            log.warning(f"watchdog: settings not loaded: {e}")
            return None

    def _tell(self, name: str, text: str) -> bool:
        settings = self.settings()
        if settings is None:
            return False
        from blink_watcher.core.health import tell
        return tell(settings, name, text)

    def _watch(self, now: float) -> None:
        """Stale health files (stuck services) and services that stay up again after crashing;
        also applies the security mode changes that are due (core.security_mode.current), so a
        scheduled or timed change happens on time even when no service reads the mode."""
        for svc in self.services:
            if svc.running and svc.crash_alerted and now - svc.started_at >= STABLE_AFTER:
                svc.quick_exits, svc.crash_alerted = 0, False
                self._tell(f"{svc.name.upper()}_RUNNING", f"Fixed: the {svc.name} service is running again.")
        settings = self.settings()
        if settings is not None:
            from blink_watcher.core import security_mode
            security_mode.current(settings)
        stale_min = _setting(settings, "health_stale_minutes", 0)
        if stale_min <= 0:
            return
        from blink_watcher.core.health import read_all
        health = read_all(settings)
        for svc in self.services:
            if svc.name not in HEARTBEAT_SERVICES or not svc.running or svc.reload_by is not None:
                continue
            updated = _epoch((health.get(svc.name) or {}).get("updated_at"))
            last = max(updated or 0.0, svc.started_wall)
            if time.time() - last < stale_min * 60:
                continue
            since = datetime.fromtimestamp(last).strftime("%H:%M")
            log.error(f"{svc.name} has not reported since {since} (over {stale_min} min) - it is stuck; restarting it",
                      extra={"alert": False})  # told just below
            self._tell(f"{svc.name.upper()}_STUCK", f"Problem: the {svc.name} service was stuck (no check since "
                                                    f"{since}). Restarted it.")
            svc.reload_by = now  # killed and started again at once, like a settings change

    # ---------- main loop ----------

    def run(self) -> int:
        log.info(f"Starting {', '.join(s.name for s in self.services)} (Ctrl+C stops; "
                 f"settings changes in {ENV_FILE.name} restart the services they affect)")
        self.env_mtime, self.env_values = self._env_state()
        self.env_changed_at: Optional[float] = None
        for svc in self.services:
            self.start(svc)
        watched_at, wall = time.monotonic(), time.time()
        from blink_watcher.control import stop_file
        stop_request = stop_file(_settings)
        try:
            while True:
                time.sleep(0.5)
                if stop_request.exists():
                    log.info("Stop requested (vdemo stop)")
                    raise KeyboardInterrupt
                now = time.monotonic()
                if time.time() - wall > ASLEEP_SEC:
                    log.info(f"Computer was asleep for {(time.time() - wall) / 60:.0f} min - "
                             f"the services get time to report again")
                    for svc in self.services:
                        svc.started_wall = time.time()
                wall = time.time()
                self._check_settings(now)
                if now - watched_at >= WATCHDOG_SEC:
                    watched_at = now
                    try:
                        self._watch(now)
                    except Exception:  # noqa: BLE001 - the watchdog must never stop the supervisor
                        log.exception("watchdog check failed")
                for svc in self.services:
                    if svc.reload_by is not None and svc.running and now >= svc.reload_by:
                        log.warning(f"{svc.name} did not stop in {STOP_TIMEOUT:g}s; killed")
                        self._kill(svc)
                    if svc.proc is not None and svc.proc.poll() is not None:
                        if svc.reader:
                            svc.reader.join(timeout=2)  # its last lines first
                        self._exited(svc, now)
                    elif svc.proc is None and svc.restart_at is not None and now >= svc.restart_at:
                        self.start(svc)
        except KeyboardInterrupt:
            try:
                self.stop_all()
            except KeyboardInterrupt:  # Ctrl+C again: don't wait, and leave nothing running
                log.warning("Ctrl+C again - killing the services that are still running")
                self.kill_all()
            log.info("All services stopped")
            return 0


def _setting(settings, name: str, default: int) -> int:
    return getattr(settings, name, default) if settings is not None else default


def _epoch(iso: Optional[str]) -> Optional[float]:
    try:
        return datetime.fromisoformat(iso).timestamp() if iso else None
    except ValueError:
        return None


def affected(keys: list[str]) -> set[str]:
    """The services that read these options: the "@programs" of their groups
    in .env.example; every service for an option not documented there."""
    from blink_watcher.settings.document import parse_example
    owners = {o.key: set(g.programs) for g in parse_example() for o in g.options}
    every = {s.name for s in default_services()}
    return set().union(*(owners.get(k) or every for k in keys))


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
    parser.add_argument("--background", action="store_true",
                        help="No console (vdemo start): the services' output stays in the log files")
    args = parser.parse_args(argv)
    chosen = select(services, _names(args.only), _names(args.skip))
    from blink_watcher.control import find
    others = find(exclude=os.getpid())
    if others:
        print(f"already running (pid {', '.join(str(p.pid) for p in others)}): "
              f"stop it first (vdemo stop, or Ctrl+C in its window)", file=sys.stderr)
        return 1
    setup_logging(PROGRAM, args.verbose, log.name)
    if not WINDOWS:  # a SIGTERM (service manager, kill) stops the children like Ctrl+C
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    return Supervisor(chosen, background=args.background).run()
