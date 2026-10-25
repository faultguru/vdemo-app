"""The security mode: whether the residents are around, set from the web console
or Telegram and read by the video processor, the describer and the chat agent
(notifications and chat) to weigh what they see.

  normal    the usual routine
  home      residents at home: clips of the indoor cameras (INDOOR_CAMERAS)
            are saved as they are, without any analysis, description or
            notification. Turns back to normal at
            HOME_MODE_UNTIL (local time) unless another end is given.
  away      residents out; stays until changed, or until its end
  vacation  residents away for longer; stays until changed, or until its end

The mode is runtime state, not configuration: STORAGE_ROOT/state/security_mode.json
({"mode", "since", "until", "by"}), read on every use, so a change applies
without a restart. Every change is appended to security_mode.history.jsonl.
A mode with an `until` turns back to normal then (by "auto").

Modes can also be scheduled ahead (schedule(): e.g. a vacation from one date
to another): STORAGE_ROOT/state/security_mode.schedule.json
({"entries": [{"id", "mode", "start", "end", "by", "note", "created_at"}]}).
An entry takes effect at its start (by "schedule", until its end) and leaves
the schedule; a manual change in between simply replaces the mode, and the
scheduled end then no longer applies. current() applies due changes - every
service reads it, and the supervisor's watchdog calls it every minute - and
Telegram is told of each change made on its own (not of HOME's nightly end).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time as _time
import uuid
from contextlib import contextmanager
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
SCHEDULE_FILE = "security_mode.schedule.json"
LOCK_FILE = "security_mode.lock"
LOCK_STALE_SEC = 30.0
BY_AUTO_AWAY = "auto-away"  # set by pipeline.notify.auto_away (the household's vehicle left / came back)
PROMPT = "security_mode.txt"   # the block for the LLM prompts, and what each mode means


@dataclass(frozen=True)
class ModeState:
    mode: str = "normal"
    since: Optional[str] = None   # ISO time it was set
    until: Optional[str] = None   # ISO time it turns back to normal
    by: str = "default"           # who set it: console | telegram | chat | schedule | auto-away | auto | default
    note: Optional[str] = None    # why, as the user put it (e.g. at the office)

    @property
    def label(self) -> str:
        return LABELS[self.mode]

    def to_dict(self) -> dict:
        return {**asdict(self), "label": self.label}

    def headline(self) -> str:
        """E.g. 'HOME until 23:00' / 'VACATION until Sun 11-02 23:59' / 'AWAY since Sat 18:40'."""
        if self.until:
            return f"{self.label.upper()} until {when(self.until, self.since)}"
        if self.since and self.mode != "normal":
            return f"{self.label.upper()} since {datetime.fromisoformat(self.since):%a %H:%M}"
        return self.label.upper()


@dataclass(frozen=True)
class Scheduled:
    """A mode to set later: from `start` (ISO local time) to `end` (None: until changed)."""
    id: str
    mode: str
    start: str
    end: Optional[str] = None
    by: str = "chat"
    note: Optional[str] = None
    created_at: Optional[str] = None

    def describe(self) -> str:
        """E.g. 'VACATION from Sat 10-25 00:00 to Sun 11-02 23:59 (family trip) [id 3f2a]'."""
        text = f"{LABELS[self.mode].upper()} from {when(self.start)}"
        text += f" to {when(self.end, self.start)}" if self.end else " until changed"
        return text + (f" ({self.note})" if self.note else "") + f" [id {self.id}]"


def when(iso: str, ref: Optional[str] = None) -> str:
    """HH:MM if on the same day as `ref` (default: now), else 'Sun 11-02 23:59'."""
    t = datetime.fromisoformat(iso)
    base = datetime.fromisoformat(ref).date() if ref else datetime.now(t.tzinfo).date()
    return f"{t:%H:%M}" if t.date() == base else f"{t:%a %m-%d %H:%M}"


def _paths(settings: "Settings") -> tuple[Path, Path]:
    folder = state_dir(settings)
    return folder / STATE_FILE, folder / HISTORY_FILE


@contextmanager
def _locked(settings: "Settings"):
    """One writer of the mode and its schedule at a time, across the services."""
    lock = state_dir(settings) / LOCK_FILE
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if _time.time() - lock.stat().st_mtime > LOCK_STALE_SEC:
                    log.warning(f"Removing stale lock {lock}")
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            _time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _now(settings: "Settings") -> datetime:
    return datetime.now(ZoneInfo(settings.timezone))


def _write(path: Path, data: dict) -> None:
    """Atomic: readers in the other services never see a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _save(settings: "Settings", state: ModeState) -> ModeState:
    path, history = _paths(settings)
    _write(path, asdict(state))
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(state)) + "\n")
    log.info(f"Security mode set to {state.headline()} (by {state.by})")
    return state


def home_until(settings: "Settings", now: datetime) -> datetime:
    """The next HOME_MODE_UNTIL (HH:MM local time) after `now`."""
    hour, minute = (int(x) for x in settings.home_mode_until.split(":"))
    end = datetime.combine(now.date(), time(hour, minute), tzinfo=now.tzinfo)
    return end if end > now else end + timedelta(days=1)


def _read_state(settings: "Settings") -> ModeState:
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
    return state


def _read_schedule(settings: "Settings") -> list[Scheduled]:
    path = state_dir(settings) / SCHEDULE_FILE
    try:
        entries = json.loads(path.read_text(encoding="utf-8")).get("entries") or []
        return sorted((Scheduled(**e) for e in entries), key=lambda e: e.start)
    except FileNotFoundError:
        return []
    except (ValueError, TypeError, AttributeError) as e:
        log.warning(f"Unreadable {path} ({e}) - no scheduled security modes")
        return []


def _write_schedule(settings: "Settings", entries: list[Scheduled]) -> None:
    _write(state_dir(settings) / SCHEDULE_FILE, {"entries": [asdict(e) for e in sorted(entries, key=lambda e: e.start)]})


def _due(state: ModeState, entries: list[Scheduled], now: datetime) -> bool:
    return bool(state.until and now >= datetime.fromisoformat(state.until)) or any(
        now >= datetime.fromisoformat(e.start) for e in entries)


def current(settings: "Settings") -> ModeState:
    """The mode now, after applying what is due: a mode past its `until` turns back
    to normal, and a scheduled mode whose start has come takes effect (both saved)."""
    now = _now(settings)
    state = _read_state(settings)
    if not _due(state, _read_schedule(settings), now):
        return state
    with _locked(settings):
        state, entries = _read_state(settings), _read_schedule(settings)  # another service may have done it
        if not _due(state, entries, now):
            return state
        ended = state if state.until and now >= datetime.fromisoformat(state.until) else None
        if ended:
            state = _save(settings, ModeState("normal", ended.until, None, "auto"))
        due = [e for e in entries if now >= datetime.fromisoformat(e.start)]
        _write_schedule(settings, [e for e in entries if e not in due])
        live = [e for e in due if not e.end or now < datetime.fromisoformat(e.end)]
        for e in due:
            if e not in live:
                log.info(f"Scheduled security mode skipped, its time has passed: {e.describe()}")
        started = live[-1] if live else None  # the latest start wins
        if started:
            state = _save(settings, ModeState(started.mode, started.start, started.end, "schedule", started.note))
    if started:
        _tell(settings, "MODE_SCHEDULED", f"Security mode is now {state.headline()} (scheduled"
                                          + (f": {started.note}" if started.note else "") + ").")
    elif ended and ended.mode != "home":  # HOME ends every night: not worth a message
        _tell(settings, "MODE_ENDED", f"Security mode is back to NORMAL: {ended.label.upper()} ended at "
                                      f"{when(ended.until)}.")
    return state


def _tell(settings: "Settings", name: str, text: str) -> None:
    try:
        from blink_watcher.core.health import tell
        tell(settings, name, text)
    except Exception as e:  # noqa: BLE001 - the mode change stands without the message
        log.warning(f"Telegram not told of the mode change: {e}")


def _parse_time(settings: "Settings", value: str, what: str) -> datetime:
    """A local time 'YYYY-MM-DDTHH:MM' (seconds and an offset allowed)."""
    try:
        t = datetime.fromisoformat(value.strip())
    except (ValueError, AttributeError) as e:
        raise ValueError(f"{what} must be a local time like 2026-10-25T08:00, got {value!r}") from e
    return t.replace(tzinfo=ZoneInfo(settings.timezone)) if t.tzinfo is None else t


def set_mode(settings: "Settings", mode: str, by: str, until: Optional[str] = None,
             note: Optional[str] = None) -> ModeState:
    """Set the mode now. `until` (local time): when it turns back to normal; HOME
    without one ends at HOME_MODE_UNTIL, the others stay until changed."""
    if mode not in MODES:
        raise ValueError(f"unknown security mode {mode!r} (known: {', '.join(MODES)})")
    now = _now(settings)
    end = _parse_time(settings, until, "until") if until and mode != "normal" else None
    if end is not None and end <= now:
        raise ValueError(f"until {end:%Y-%m-%d %H:%M} is not in the future")
    if end is None and mode == "home":
        end = home_until(settings, now)
    with _locked(settings):
        return _save(settings, ModeState(mode, now.isoformat(timespec="seconds"),
                                         end.isoformat(timespec="seconds") if end else None, by, note or None))


def schedule(settings: "Settings", mode: str, start: str, end: Optional[str] = None, by: str = "chat",
             note: Optional[str] = None) -> Scheduled:
    """Schedule `mode` from `start` to `end` (local times; no end: until changed).
    Raises ValueError for a bad mode / time or an overlap with another entry."""
    if mode not in MODES:
        raise ValueError(f"unknown security mode {mode!r} (known: {', '.join(MODES)})")
    now = _now(settings)
    t0 = _parse_time(settings, start, "start")
    t1 = _parse_time(settings, end, "end") if end else None
    if t1 is not None and t1 <= max(t0, now):
        raise ValueError(f"end {t1:%Y-%m-%d %H:%M} must be after the start and in the future")
    with _locked(settings):
        entries = _read_schedule(settings)
        for e in entries:
            e0, e1 = datetime.fromisoformat(e.start), datetime.fromisoformat(e.end) if e.end else None
            if (t1 is None or t1 > e0) and (e1 is None or e1 > t0):
                raise ValueError(f"overlaps the scheduled {e.describe()}; cancel it first")
        entry = Scheduled(uuid.uuid4().hex[:6], mode, t0.isoformat(timespec="seconds"),
                          t1.isoformat(timespec="seconds") if t1 else None, by, note or None,
                          now.isoformat(timespec="seconds"))
        _write_schedule(settings, entries + [entry])
    log.info(f"Security mode scheduled: {entry.describe()} (by {by})")
    current(settings)  # a start that has already come applies at once
    return entry


def cancel(settings: "Settings", entry_id: str) -> Scheduled:
    with _locked(settings):
        entries = _read_schedule(settings)
        match = [e for e in entries if e.id == entry_id.strip()]
        if not match:
            raise ValueError(f"no scheduled security mode with id {entry_id!r}")
        _write_schedule(settings, [e for e in entries if e.id != entry_id.strip()])
    log.info(f"Scheduled security mode cancelled: {match[0].describe()}")
    return match[0]


def upcoming(settings: "Settings") -> list[Scheduled]:
    """The scheduled modes, soonest first."""
    return _read_schedule(settings)


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


def prompt_block(state: ModeState, scheduled: tuple[Scheduled, ...] = ()) -> str:
    """The security mode for an LLM prompt: the mode, since / until, what it means
    (+ the scheduled changes, for the chat agent)."""
    template, meanings = _prompt_sections()
    span = (f"since {datetime.fromisoformat(state.since):%A %H:%M}" if state.since else "the default")
    if state.until:
        span += f", until {datetime.fromisoformat(state.until):%A %Y-%m-%d %H:%M}, then normal"
    if state.note:
        span += f"; {state.note}"
    block = template.format(mode=state.label.upper(), when=span, meaning=meanings.get(state.mode, ""))
    if scheduled:
        block += "\nScheduled changes:\n" + "\n".join(f"- {e.describe()}" for e in scheduled)
    return block
