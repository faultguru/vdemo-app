"""The security mode: whether the residents are around, set from the web console
or Telegram and read by the video processor, the describer and the chat agent
(notifications and chat) to weigh what they see.

  normal    the usual routine
  home      residents at home: clips of the indoor cameras (INDOOR_CAMERAS)
            are saved as they are, without any analysis, description or
            notification. Turns back to normal at
            HOME_MODE_UNTIL (local time).
  away      residents out; stays until changed
  vacation  residents away for longer; stays until changed

The mode is runtime state, not configuration: STORAGE_ROOT/state/security_mode.json
({"mode", "since", "until", "by"}), read on every use, so a change applies
without a restart. Every change is appended to security_mode.history.jsonl.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, get_args
from zoneinfo import ZoneInfo

from blink_watcher.core.prompts import load_prompt
from blink_watcher.core.runtime import state_dir

if TYPE_CHECKING:
    from blink_watcher.settings import Settings

log = logging.getLogger(__name__)

SecurityMode = Literal["normal", "home", "away", "vacation"]
MODES: tuple[str, ...] = get_args(SecurityMode)
LABELS = {"normal": "Normal", "home": "Home", "away": "Away", "vacation": "Vacation"}

STATE_FILE = "security_mode.json"
HISTORY_FILE = "security_mode.history.jsonl"
PROMPT = "security_mode.txt"   # the block for the LLM prompts, and what each mode means


@dataclass(frozen=True)
class ModeState:
    mode: str = "normal"
    since: Optional[str] = None   # ISO time it was set
    until: Optional[str] = None   # ISO time it turns back to normal (home only)
    by: str = "default"           # who set it: console | telegram | auto | default

    @property
    def label(self) -> str:
        return LABELS[self.mode]

    def to_dict(self) -> dict:
        return {**asdict(self), "label": self.label}

    def headline(self) -> str:
        """E.g. 'HOME until 23:00' / 'AWAY since Sat 18:40'."""
        if self.until:
            return f"{self.label.upper()} until {datetime.fromisoformat(self.until):%H:%M}"
        if self.since and self.mode != "normal":
            return f"{self.label.upper()} since {datetime.fromisoformat(self.since):%a %H:%M}"
        return self.label.upper()


def _paths(settings: "Settings") -> tuple[Path, Path]:
    folder = state_dir(settings)
    return folder / STATE_FILE, folder / HISTORY_FILE


def _now(settings: "Settings") -> datetime:
    return datetime.now(ZoneInfo(settings.timezone))


def _write(path: Path, state: ModeState) -> None:
    """Atomic: readers in the other services never see a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)
    os.replace(tmp, path)


def _save(settings: "Settings", state: ModeState) -> ModeState:
    path, history = _paths(settings)
    _write(path, state)
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(state)) + "\n")
    log.info(f"Security mode set to {state.headline()} (by {state.by})")
    return state


def home_until(settings: "Settings", now: datetime) -> datetime:
    """The next HOME_MODE_UNTIL (HH:MM local time) after `now`."""
    hour, minute = (int(x) for x in settings.home_mode_until.split(":"))
    end = datetime.combine(now.date(), time(hour, minute), tzinfo=now.tzinfo)
    return end if end > now else end + timedelta(days=1)


def current(settings: "Settings") -> ModeState:
    """The mode now. A home mode past its `until` is turned back to normal (and saved)."""
    path, _ = _paths(settings)
    try:
        state = ModeState(**json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return ModeState()
    except (ValueError, TypeError) as e:
        log.warning(f"Unreadable {path} ({e}) - security mode taken as normal")
        return ModeState()
    if state.mode not in MODES:
        log.warning(f"Unknown security mode {state.mode!r} in {path} - taken as normal")
        return ModeState()
    now = _now(settings)
    if state.until and now >= datetime.fromisoformat(state.until):
        return _save(settings, ModeState("normal", state.until, None, "auto"))
    return state


def set_mode(settings: "Settings", mode: str, by: str) -> ModeState:
    if mode not in MODES:
        raise ValueError(f"unknown security mode {mode!r} (known: {', '.join(MODES)})")
    now = _now(settings)
    until = home_until(settings, now).isoformat(timespec="seconds") if mode == "home" else None
    return _save(settings, ModeState(mode, now.isoformat(timespec="seconds"), until, by))


# ---------- prompts ----------

def _prompt_sections() -> tuple[str, dict[str, str]]:
    """prompts/security_mode.txt: the block template, then a "[mode]" section per mode."""
    template, meanings, name = [], {}, None
    for line in load_prompt(PROMPT).splitlines():
        if line.startswith("[") and line.rstrip().endswith("]"):
            name = line.strip()[1:-1]
            meanings[name] = []
        elif name is None:
            template.append(line)
        else:
            meanings[name].append(line)
    return "\n".join(template).strip(), {k: "\n".join(v).strip() for k, v in meanings.items()}


def prompt_block(state: ModeState) -> str:
    """The security mode for an LLM prompt: the mode, since / until, what it means."""
    template, meanings = _prompt_sections()
    when = (f"since {datetime.fromisoformat(state.since):%A %H:%M}" if state.since else "the default")
    if state.until:
        when += f", until {datetime.fromisoformat(state.until):%H:%M}"
    return template.format(mode=state.label.upper(), when=when, meaning=meanings.get(state.mode, ""))
