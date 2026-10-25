"""Start, stop and restart the services in the background (the `vdemo` script).

    python -m blink_watcher control start     # `run` in the background; returns once it is up
    python -m blink_watcher control stop      # stops it the Ctrl+C way, killed if it does not stop
    python -m blink_watcher control restart   # stop, then start
    python -m blink_watcher control status    # running or not (exit code 0 / 3)

`run` (blink_watcher.supervisor) is found by its command line, so a console
one (run.bat, `vdemo console`) is seen and stopped too. start runs it with no
window and its own process group, with --background: its output is only in
the log files (STORAGE_ROOT/logs/run.log and each service's). stop asks it
to stop through a request file (state/run.stop) that it checks twice a
second, and waits CONTROL_STOP_WAIT_SEC; one still running then (or one
too old to know the request file) is killed with its services.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Optional

import psutil

from blink_watcher.core.paths import PROJECT_ROOT
from blink_watcher.core.runtime import logs_dir, state_dir
from blink_watcher.settings import Settings, load_settings

WINDOWS = os.name == "nt"
STOP_FILE = "run.stop"  # in state/: a request for the running supervisor to stop


def stop_file(settings: Settings):
    return state_dir(settings) / STOP_FILE


def _is_run(cmdline: list[str]) -> bool:
    """`python -m blink_watcher run ...` (the supervisor)."""
    for i, arg in enumerate(cmdline[:-2]):
        if arg == "-m" and cmdline[i + 1] == "blink_watcher" and cmdline[i + 2] == "run":
            return True
    return False


def find(exclude: Optional[int] = None) -> list[psutil.Process]:
    """The running supervisors, without `exclude` and its launcher (a venv's
    python.exe on Windows starts the real one as its child with the same
    command line: only the outermost of such a pair is listed)."""
    skip = set()
    if exclude is not None:
        me = psutil.Process(exclude)
        skip = {exclude} | {p.pid for p in me.parents()}
    found = []
    for p in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            if p.info["pid"] not in skip and _is_run(p.info["cmdline"] or []):
                found.append(p)
        except (psutil.Error, TypeError):
            continue
    pids = {p.pid for p in found}
    return [p for p in found if p.info["ppid"] not in pids]


def _describe(procs: list[psutil.Process]) -> str:
    return ", ".join(f"pid {p.pid}, since {time.strftime('%H:%M:%S', time.localtime(p.create_time()))}"
                     for p in procs)


def _tail(settings: Settings, n: int = 15) -> str:
    try:
        lines = (logs_dir(settings) / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def start(settings: Settings) -> int:
    running = find()
    if running:
        print(f"already running ({_describe(running)})")
        return 0
    stop_file(settings).unlink(missing_ok=True)  # a request left over must not stop the new one
    args = [sys.executable, "-m", "blink_watcher", "run", "--background"]
    if WINDOWS:
        # A hidden console of its own (not DETACHED_PROCESS): the services need one for Ctrl+Break
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    else:
        flags = {"start_new_session": True}
    proc = subprocess.Popen(args, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, close_fds=True, **flags)
    deadline = time.monotonic() + settings.control_start_wait_sec
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if proc.poll() is not None:
            print(f"failed to start (exit code {proc.returncode}); the end of logs/run.log:\n{_tail(settings)}",
                  file=sys.stderr)
            return 1
        if _webui_up(settings):
            break
    print(f"started (pid {proc.pid}); web console http://{settings.webui_host}:{settings.webui_port}, "
          f"logs in {logs_dir(settings)}")
    return 0


def _webui_up(settings: Settings) -> bool:
    import socket
    try:
        with socket.create_connection((settings.webui_host, settings.webui_port), timeout=0.5):
            return True
    except OSError:
        return False


def _kill_tree(p: psutil.Process) -> None:
    try:
        procs = p.children(recursive=True) + [p]
    except psutil.Error:
        return
    for q in procs:
        try:
            q.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)


def stop(settings: Settings) -> int:
    running = find()
    if not running:
        print("not running")
        return 0
    print(f"stopping ({_describe(running)})...")
    request = stop_file(settings)
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(str(os.getpid()), encoding="utf-8")
    try:
        gone, alive = psutil.wait_procs(running, timeout=settings.control_stop_wait_sec)
        for p in alive:
            print(f"pid {p.pid} did not stop in {settings.control_stop_wait_sec}s; killing it and its services")
            _kill_tree(p)
    finally:
        request.unlink(missing_ok=True)
    print("stopped")
    return 0


def status(settings: Settings) -> int:
    running = find()
    if running:
        print(f"running ({_describe(running)}); web console http://{settings.webui_host}:{settings.webui_port}")
        return 0
    print("not running")
    return 3  # as LSB init scripts: program is not running


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["start", "stop", "restart", "status"])
    args = parser.parse_args(argv)
    settings = load_settings()
    if args.action == "start":
        return start(settings)
    if args.action == "stop":
        return stop(settings)
    if args.action == "restart":
        return stop(settings) or start(settings)
    return status(settings)
