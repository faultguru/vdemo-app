"""Chat state in the chat area (STORAGE_ROOT/CHAT_PREFIX, e.g. data-storage/chat):

    memory/memory.md                          long-term memory (chat.memory.Memory)
    events-inbox/<stamp>-<CAMERA>.event.json  events queued by the video processor
    <YYYY-MM-DD>/memory.md                    day notes (chat.memory.DayNotes)
    <YYYY-MM-DD>/events.json                  that day's events, by event_id, each with
                                              the chat agent's "notification" decision
    <YYYY-MM-DD>/chat_history.<chat id>.json  chat turns, sliding window
    outbox/<stamp>-<CAMERA>.message.json      messages waiting to be sent:
                                              {event_id, date, text, photo}
    .notify_state.json                        when media was last offered

Used by the notifier (pipeline.notify: writes events and outgoing messages),
by the video processor (a clip saved without analysis goes straight into its
day's events), by the chat bot (record_sent: a message into the chat history
once it is sent) and by the chat agent and its tools (read them back in chat).
Those services are separate processes: a day's events index is only changed
under its lock file (<date>/events.json.lock).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

MEMORY_FILE = "memory/memory.md"
DAY_NOTES = "{date}/memory.md"
EVENTS_INDEX = "{date}/events.json"
HISTORY_FILE = "{date}/chat_history.{chat_id}.json"
NOTIFY_STATE = ".notify_state.json"
OUTBOX = "outbox"
EVENTS_INBOX = "events-inbox"
MESSAGE_SUFFIX = ".message.json"

HISTORY_WINDOW = 10
_DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def local_now(tz_name: str) -> datetime:
    """Now in the configured timezone (the system's if the name is not a valid zone)."""
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:  # noqa: BLE001
        return datetime.now().astimezone()


# ---------- events index ----------

def events_index(chat_dir: Path, date: str) -> Path:
    return chat_dir / EVENTS_INDEX.format(date=date)


def event_dates(chat_dir: Path) -> list[str]:
    """Dates that have an events index, oldest first."""
    if not chat_dir.exists():
        return []
    return sorted(p.name for p in chat_dir.iterdir()
                  if p.is_dir() and _DATE_DIR.match(p.name) and (p / "events.json").exists())


def load_events(chat_dir: Path, date: str) -> list[dict]:
    path = events_index(chat_dir, date)
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("could not parse %s — starting a new index", path)
        return []
    return loaded if isinstance(loaded, list) else []


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def event_date(event_id: str) -> str:
    """The local date in an event id — "<stamp>-<CAMERA>" (e.g. 2026-10-17-11-22-04-FRONT_DOOR)
    or the older "<stamp>-00"."""
    m = _DATE.search(event_id)
    if not m:
        raise ValueError(f"no date in event id {event_id!r}")
    return m.group(0)


LOCK_STALE_SEC = 30.0  # a lock file older than this was left by a crashed writer


@contextmanager
def _index_lock(chat_dir: Path, date: str):
    """The day's events index for one writer at a time, across processes."""
    lock = events_index(chat_dir, date).with_suffix(".json.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_SEC:
                    log.warning(f"Removing stale lock {lock}")
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _store_event(chat_dir: Path, date: str, entry: dict) -> tuple[Path, int]:
    events = [e for e in load_events(chat_dir, date) if e.get("event_id") != entry["event_id"]]
    events.append(entry)
    events.sort(key=lambda e: (e.get("recorded_at") or "", e.get("event_id") or ""))
    path = events_index(chat_dir, date)
    path.write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, len(events)


def add_event(chat_dir: Path, entry: dict) -> tuple[Path, int]:
    """Add (or replace, by event_id) an event in its day's index, kept sorted
    by recording time (all cameras in one list). Returns (index path, entries in it)."""
    date = event_date(entry["event_id"])
    with _index_lock(chat_dir, date):
        return _store_event(chat_dir, date, entry)


def find_event(chat_dir: Path, event_id: str) -> Optional[dict]:
    return next((e for e in load_events(chat_dir, event_date(event_id)) if e.get("event_id") == event_id), None)


def update_event(chat_dir: Path, event_id: str, **fields) -> None:
    """Set top-level fields of an event already in its day's index."""
    date = event_date(event_id)
    with _index_lock(chat_dir, date):
        entry = find_event(chat_dir, event_id)
        if entry is None:
            raise KeyError(f"event {event_id} is not in {events_index(chat_dir, date)}")
        _store_event(chat_dir, date, {**entry, **fields})


# ---------- media-offer cooldown ----------

OFFER_COOLDOWN_SECONDS = 3600  # only invite the user to request media once per hour


def should_offer_media(chat_dir: Path) -> bool:
    """True at most once per OFFER_COOLDOWN_SECONDS (and records the offer)."""
    state_file = chat_dir / NOTIFY_STATE
    now = datetime.now().timestamp()
    try:
        last = float(json.loads(state_file.read_text(encoding="utf-8")).get("last_offer_ts") or 0)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        last = 0.0
    if now - last < OFFER_COOLDOWN_SECONDS:
        return False
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"last_offer_ts": now}), encoding="utf-8")
    return True


# ---------- chat history ----------

class ChatHistory:
    """Per-chat sliding window of the last HISTORY_WINDOW text turns, one file
    per day. Loading reads the newest day file on or before the given date,
    so the conversation carries over midnight; saving writes that date's file."""

    def __init__(self, chat_dir: Path, tz_name: str):
        self.chat_dir = chat_dir
        self.tz_name = tz_name
        self._lock = Lock()

    def today(self) -> str:
        return local_now(self.tz_name).strftime("%Y-%m-%d")

    def path(self, chat_id: str, date: str | None = None) -> Path:
        return self.chat_dir / HISTORY_FILE.format(date=date or self.today(), chat_id=chat_id)

    def load(self, chat_id: str, date: str | None = None) -> list[dict]:
        date = date or self.today()
        name = PurePosixPath(HISTORY_FILE.format(date="x", chat_id=chat_id)).name
        days = sorted((p for p in self.chat_dir.glob(f"*/{name}")
                       if _DATE_DIR.match(p.parent.name) and p.parent.name <= date),
                      key=lambda p: p.parent.name) if self.chat_dir.exists() else []
        if not days:
            return []
        try:
            return json.loads(days[-1].read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.warning("corrupt history %s, resetting", days[-1])
            return []

    def save(self, chat_id: str, history: list[dict], date: str | None = None) -> None:
        path = self.path(chat_id, date)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history[-HISTORY_WINDOW:], ensure_ascii=False, indent=2), encoding="utf-8")

    def append(self, chat_id: str, turn: dict, date: str | None = None) -> None:
        with self._lock:
            history = self.load(chat_id, date)
            history.append(turn)
            self.save(chat_id, history, date)

    def clear(self, chat_id: str) -> int:
        """Start today's window empty (earlier days' files are kept). Returns
        the number of turns that were in the window."""
        with self._lock:
            prior = len(self.load(chat_id))
            self.save(chat_id, [])
        return prior


class MemoryHistory:
    """ChatHistory's interface, kept in memory only (the web console's chat):
    the same sliding window, nothing written to disk, gone on restart."""

    def __init__(self):
        self._turns: dict[str, list[dict]] = {}

    def path(self, chat_id: str, date: str | None = None) -> None:
        return None

    def load(self, chat_id: str, date: str | None = None) -> list[dict]:
        return list(self._turns.get(chat_id, []))

    def save(self, chat_id: str, history: list[dict], date: str | None = None) -> None:
        self._turns[chat_id] = list(history[-HISTORY_WINDOW:])

    def clear(self, chat_id: str) -> int:
        return len(self._turns.pop(chat_id, []))


# ---------- outbox ----------

def queue_message(chat_dir: Path, name: str, message: dict) -> Path:
    """Write an outgoing message to outbox/<name>.message.json; it stays there
    until the sender has sent it (see record_sent)."""
    path = chat_dir / OUTBOX / f"{name}{MESSAGE_SUFFIX}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(message, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def queued_messages(chat_dir: Path) -> list[Path]:
    """Outgoing messages, oldest first."""
    outbox = chat_dir / OUTBOX
    return sorted(outbox.glob(f"*{MESSAGE_SUFFIX}")) if outbox.exists() else []


def record_sent(chat_dir: Path, tz_name: str, chat_id: str, message_path: Path, photo_sent: bool = False) -> None:
    """Call once a queued message has been sent: the message goes into the
    chat history (on its event's day; "with picture" if the photo went with
    it), the event's notification gets "sent_at", and the message leaves the
    outbox."""
    message = json.loads(message_path.read_text(encoding="utf-8"))
    ChatHistory(chat_dir, tz_name).append(
        chat_id, notification_turn(message["text"], message.get("event_id"), photo_sent), date=message.get("date"))
    event_id = message.get("event_id")
    entry = find_event(chat_dir, event_id) if event_id else None
    if entry is not None:
        sent_at = datetime.now().astimezone().isoformat(timespec="seconds")
        update_event(chat_dir, event_id, notification={**(entry.get("notification") or {}), "sent_at": sent_at})
    message_path.unlink(missing_ok=True)


def notification_turn(text: str, event_id: str | None = None, photo: bool = False) -> dict:
    """A server-initiated notification as a model turn in the chat history."""
    tag = "[auto-notification" + (f" event_id={event_id}" if event_id else "") + (" with picture" if photo else "") + "]"
    return {"role": "model", "text": f"{tag} {text}".strip()}


MEDIA_TOOLS = {"send_thumbnail": "picture", "send_video": "video"}


def media_turn_text(tool_calls: list[dict]) -> str:
    """"[sent video event_id=...]" lines for the pictures / videos a reply sent
    — so the history shows media that went out without any text."""
    return "\n".join(
        f"[sent {MEDIA_TOOLS[c['name']]} event_id={c['args'].get('event_id')}]"
        for c in tool_calls if c["name"] in MEDIA_TOOLS and "error" not in c["result"]
    )
