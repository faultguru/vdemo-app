"""The command line: one entry point for every program.

    python -m blink_watcher <command> [options]
    python -m blink_watcher <command> --help     # the command's own options
    python -m blink_watcher --help               # the commands

Each command is a module with main(argv) -> exit code, imported only when that
command runs, so e.g. `webui --help` does not load OpenCV or YOLO.

On Windows, Ctrl+Break stops a command like Ctrl+C does (KeyboardInterrupt):
that is how `run` (blink_watcher.supervisor) stops its children.
"""
from __future__ import annotations

import importlib
import signal
import sys
from dataclasses import dataclass
from typing import Optional

PROG = "python -m blink_watcher"


@dataclass(frozen=True)
class Command:
    module: str
    help: str


COMMANDS: dict[str, Command] = {
    "run": Command("blink_watcher.supervisor",
                   "every service in one console: fetch, process, notify, chat-bot, webui"),
    "fetch": Command("blink_watcher.pipeline.fetch", "download new clips from the camera provider"),
    "process": Command("blink_watcher.pipeline.process",
                       "analyze queued clips (motion, YOLO, trim) and describe them with the LLM"),
    "notify": Command("blink_watcher.pipeline.notify",
                      "decide on each new event's notification (chat agent) and queue it"),
    "chat-bot": Command("blink_watcher.chat.bot", "Telegram bot: sends notifications, answers messages"),
    "webui": Command("blink_watcher.webui.server", "local web console: events, live monitor, chat, settings"),
    "stitch": Command("blink_watcher.stitch.stitcher", "show a day's activities (related clips grouped)"),
    "describe": Command("blink_watcher.describe.cli", "describe a clip's key frames with the LLM"),
    "frames": Command("blink_watcher.vision.sampling", "sample and trim a clip's frames (no AI)"),
    "hwmonitor": Command("blink_watcher.tools.hwmonitor", "log GPU / CPU / RAM use for diagnostics"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"usage: {PROG} <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {cmd.help}" for name, cmd in COMMANDS.items()]
    lines += ["", f"{PROG} <command> --help shows a command's options."]
    return "\n".join(lines)


def run_command(name: str, argv: list[str]) -> int:
    module = importlib.import_module(COMMANDS[name].module)
    # argparse names the program after argv[0]: show the command as typed
    sys.argv[0] = f"{PROG} {name}"
    return module.main(argv) or 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"{PROG}: unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    return run_command(name, rest)
