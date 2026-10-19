"""The notifier service: hands the events queued by the video processor to
the chat agent, which decides whether to notify the user and writes the message.

Reads chat/events-inbox/<stamp>-<CAMERA>.event.json, oldest recording first, and
handles up to EVENT_PROCESS_LIMIT of them (0 = all). For each event:

  1. record   — the event is added to the chat agent's daily event list,
                chat/<YYYY-MM-DD>/events.json (an entry with the same
                event_id is replaced).
  2. decide   — ChatAgent.notify(), the same agent and tool loop the chat bot
                uses (CHAT_AGENT_MODEL / CHAT_AGENT_EFFORT on LLM_PROVIDER),
                with prompts/chat_notify.system.txt, the
                long-term memory (chat/memory/memory.md) and the event day's
                notes (chat/<date>/memory.md) and the day's earlier events. It
                decides send or skip — what the user asked to hear about or
                not, earlier alerts, routine activity — and writes the message;
                it has no tools and never changes memory. The request of each
                round (round-<n>/) and the result (RESULT_FILE: the agent's
                reply, answer fields, usage) go to the clip's work folder in
                the temp area, <TEMP_VIDEOS>/<YYYY-MM-DD>/<Camera_Name>/<stamp>/event/.
     link     — the event is stitched to the day's earlier events with the
                same subject ("track" in the event list).
     close    — once a message is queued, the one-off requests of the day
                notes it fulfils ("alert-on:") become "alert-done:", so the
                same request does not alert again.
  3. queue    — the decision is stored with the event in the daily event
                list ("notification": {decision, tone, message, skip_reason,
                model, composed_at, message_file, sent_at}); only if a
                message is to be sent, chat/outbox/<stamp>-<CAMERA>.message.json
                is written — just the message: {event_id, date, text, photo
                (the event's thumbnail)}. The chat bot sends it and then adds
                it to the chat history (chat.store.record_sent).
  4. clean up — the inbox file and the event/ work folder are removed.

--preserve (testing): the inbox file and event/ work folder stay, so the same
events can be processed again. The event list entry and outbox message are
replaced on each run.
A failed event keeps its inbox file and is retried next run.

Usage:
    python -m blink_watcher notify                        # EVENT_PROCESS_LIMIT from config
    python -m blink_watcher notify --limit 0              # all queued events
    python -m blink_watcher notify --limit 1 --preserve   # testing: keep inbox + LLM request/response
    python -m blink_watcher notify --daemon               # keep running, poll the inbox every second
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

from blink_watcher.chat import store as chat_store
from blink_watcher.chat.agent import ChatAgent, Notification
from blink_watcher.chat.tools import EventTools
from blink_watcher.core.daemon import run_daemon
from blink_watcher.core.logs import setup_logging
from blink_watcher.core.storage import CHAT, TEMP_VIDEOS, Storage, make_storage
from blink_watcher.describe import RESULT_FILE
from blink_watcher.settings import Settings, load_settings

log = logging.getLogger("blink_watcher.pipeline.notify")

PROGRAM = "notify"
INBOX = chat_store.EVENTS_INBOX
WORK = "event"  # subfolder of the clip's work folder in the temp area
EVENT_SUFFIX = ".event.json"


# ---------- Run context ----------

@dataclass
class Options:
    limit: int      # events per run (per batch with --daemon), 0 = no limit
    preserve: bool  # keep inbox file + event/ work folder
    daemon: bool = False  # keep polling the inbox


@dataclass
class Areas:
    chat: Storage   # events-inbox/, outbox/, daily event lists
    temp: Storage   # per-clip work folders

    @property
    def chat_dir(self) -> Path:
        return self.chat.local_path("")


@dataclass
class Event:
    """One inbox entry and the keys derived from it."""
    inbox_key: str  # events-inbox/<stamp>-<CAMERA>.event.json in the chat area
    body: dict      # the queued .event.json

    @property
    def name(self) -> str:
        """<stamp>-<CAMERA>"""
        return PurePosixPath(self.inbox_key).name.removesuffix(EVENT_SUFFIX)

    @property
    def event_id(self) -> str:
        return self.body["event_id"]

    @property
    def date(self) -> str:
        return chat_store.event_date(self.event_id)

    @property
    def work_key(self) -> str:
        """<YYYY-MM-DD>/<Camera_Name>/<stamp>/event in the temp area — inside
        the clip's work folder (same layout as the video processor's)."""
        link = self.body.get("processed") or self.body["original"]  # {"folder": "<area>/<date>/<camera>", "video_file"}
        folder = PurePosixPath(link["folder"]).parts[1:]
        return str(PurePosixPath(*folder, PurePosixPath(link["video_file"]).stem, WORK))


# ---------- Steps ----------

def record_event(areas: Areas, event: Event) -> None:
    """Step 1: the event into its day's event list."""
    path, count = chat_store.add_event(areas.chat_dir, event.body)
    log.info(f"  [1/4] Event recorded in {areas.chat.folder}/{path.relative_to(areas.chat_dir).as_posix()} "
             f"({count} event(s) that day)")


def decide(agent: ChatAgent, areas: Areas, event: Event) -> Optional[Notification]:
    """Step 2: the chat agent decides whether to notify and writes the
    message. None if the event has no description (nothing to tell)."""
    summary = event.body.get("summary") or {}
    if not (summary.get("description") or "").strip():
        log.info("  [2/4] Notification skipped: event has no description")
        return None
    log.info(f"  [2/4] Chat agent deciding on a notification ({agent.model}, {agent.llm.name})...")
    work_dir = areas.temp.local_path(event.work_key)
    work_dir.mkdir(parents=True, exist_ok=True)
    notification, turn, prompt = asyncio.run(agent.notify(
        event.body, work_dir, progress=lambda msg, level=logging.INFO: log.log(level, f"        {msg.strip()}"),
    ))
    (work_dir / RESULT_FILE).write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **turn.record(),
        "user_prompt": prompt,
        "reply": notification.reply,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"        Response received in {turn.rounds} round(s), {turn.duration_ms / 1000:.1f}s: "
             f"{turn.usage.input_tokens} input tokens, {turn.usage.output_tokens} output tokens"
             + (f", ~${turn.cost_usd:.4f}" if turn.cost_usd else ""))
    if notification.send:
        log.info(f"        Notification to be sent ({notification.tone or 'no tone'}): {notification.message}")
    else:
        log.info(f"        Notification suppressed: {notification.skip_reason}")
    return notification


def link_subject(areas: Areas, event: Event, notification: Optional[Notification]) -> None:
    """Step 2b: stitch the event to the day's earlier events with the same
    subject (any camera, sent or not): "track": {"id", "subject", "same_as"}.
    Linked events share the track id (the earliest linked event's track, else
    its event id), so one person / vehicle can be followed from camera to camera."""
    if notification is None or not (notification.same_subject_as or notification.subject):
        return
    earlier = {e["event_id"]: e for e in chat_store.load_events(areas.chat_dir, event.date)
               if e.get("event_id") != event.event_id}
    same_as = [i for i in dict.fromkeys(notification.same_subject_as) if i in earlier]
    unknown = [i for i in notification.same_subject_as if i not in earlier]
    if unknown:
        log.warning(f"        Unknown event id(s) in same_subject_as ignored: {', '.join(unknown)}")
    linked = sorted((earlier[i] for i in same_as), key=lambda e: e.get("recorded_at") or "")
    track_id = next(((e.get("track") or {}).get("id") for e in linked if (e.get("track") or {}).get("id")),
                    linked[0]["event_id"] if linked else event.event_id)
    chat_store.update_event(areas.chat_dir, event.event_id,
                            track={"id": track_id, "subject": notification.subject, "same_as": same_as})
    if same_as:
        cams = ", ".join(dict.fromkeys(f"{e.get('camera') or '?'} {(e.get('recorded_at') or '')[11:16]}"
                                       for e in linked))
        log.info(f"        Same subject as {len(same_as)} earlier event(s) ({cams}) - track {track_id}")
    else:
        log.info(f"        New subject: {notification.subject or '(no signature)'} - track {track_id}")


def notification_record(notification: Optional[Notification]) -> dict:
    """The chat agent's decision, as stored with the event in the daily event list."""
    if notification is None:
        return {"decision": "skip", "skip_reason": "event has no description"}
    return {
        "decision": "send" if notification.send else "skip",
        "tone": notification.tone if notification.send else None,
        "message": notification.message if notification.send else None,
        "skip_reason": None if notification.send else notification.skip_reason,
        "model": notification.models[-1] if notification.models else None,  # the model that gave the reply
        "composed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def queue_message(areas: Areas, event: Event, notification: Optional[Notification]) -> bool:
    """Step 3: the decision into the daily event list; the message (text +
    thumbnail) to chat/outbox/ only if one is to be sent — the chat bot adds
    it to the chat history once sent. Returns True if queued."""
    record = notification_record(notification)
    index = f"{areas.chat.folder}/{chat_store.EVENTS_INDEX.format(date=event.date)}"
    if record["decision"] != "send":
        chat_store.update_event(areas.chat_dir, event.event_id, notification=record)
        log.info(f"  [3/4] Message not queued: nothing to send; decision recorded in {index}")
        return False
    photo = event.body.get("thumbnail")  # relative to STORAGE_ROOT
    if not photo:
        log.warning("  [3/4] Event has no thumbnail - message queued without photo")
    path = chat_store.queue_message(areas.chat_dir, event.name, {
        "event_id": event.event_id,
        "date": event.date,
        "text": notification.message,
        "photo": photo,
    })
    message_file = path.relative_to(areas.chat_dir).as_posix()
    chat_store.update_event(areas.chat_dir, event.event_id,
                            notification={**record, "message_file": message_file, "sent_at": None})
    log.info(f"  [3/4] Message queued: {areas.chat.folder}/{message_file} (photo: {photo or 'none'}); "
             f"decision recorded in {index}; chat history is updated once it is sent")
    return True


def close_requests(agent: ChatAgent, event: Event, notification: Optional[Notification]) -> None:
    """Step 3b: the one-off requests ("alert-on:" day notes) the sent
    message fulfils are marked done (DayNotes.complete)."""
    if notification is None or not notification.send or not notification.fulfills or agent.day_notes is None:
        return
    at = (event.body.get("recorded_at") or "")[11:16]
    done = agent.day_notes.complete(notification.fulfills,
                                    f"fulfilled by the {at} {event.body.get('camera') or 'camera'} alert",
                                    date=event.date)
    if done:
        log.info(f"        Request(s) completed: {'; '.join(done)}")
    else:
        log.warning(f"        No open alert-on request matched: {'; '.join(notification.fulfills)}")


def finish(areas: Areas, event: Event, opts: Options, t_event: float) -> None:
    """Step 4: remove the inbox file and work folder (both kept with --preserve)."""
    inbox_file = f"{INBOX}/{PurePosixPath(event.inbox_key).name}"
    if opts.preserve:
        log.info(f"  [4/4] Cleanup skipped (--preserve): {inbox_file} and "
                 f"{areas.temp.folder}/{event.work_key}/ retained")
    else:
        areas.chat.delete(event.inbox_key)
        areas.temp.delete_tree(event.work_key)
        log.info(f"  [4/4] Cleanup completed: {inbox_file} and {areas.temp.folder}/{event.work_key}/ removed")
    log.info(f"  Event processed in {time.perf_counter() - t_event:.1f}s")


def process_event(agent: ChatAgent, areas: Areas, event: Event, opts: Options) -> bool:
    """Steps 1-4 for one event; True if a message was queued. If a step
    fails, the inbox file stays (the event is retried next run)."""
    t_event = time.perf_counter()
    record_event(areas, event)
    notification = decide(agent, areas, event)
    link_subject(areas, event, notification)
    queued = queue_message(areas, event, notification)
    if queued:
        close_requests(agent, event, notification)
    finish(areas, event, opts, t_event)
    return queued


# ---------- Run ----------

def read_inbox(chat: Storage) -> list[Event]:
    """Queued events, oldest recording first."""
    queued = [Event(key, chat.read_json(key)) for key in chat.list(f"{INBOX}/") if key.endswith(EVENT_SUFFIX)]
    return sorted(queued, key=lambda e: (e.event_id, e.inbox_key))


def print_header(agent: ChatAgent, areas: Areas, opts: Options, waiting: int, total: int, log_file: Path) -> None:
    log.info(f"Notifier started (log file: {log_file})")
    log.info(f"Inbox: {areas.chat.local_path(INBOX)} - {total} event(s) queued, oldest first")
    log.info(f"Limit: {opts.limit or 'none'} - {waiting} event(s) will be processed")
    log.info(f"Chat agent: {agent.model} effort {agent.effort or 'default'}; memory changes: "
             f"{agent.memory_model} effort {agent.memory_effort or 'default'} ({agent.llm.name})")
    log.info(f"Preserve: {'enabled - inbox files and work folders are retained' if opts.preserve else 'disabled'}")
    log.info(f"Daemon: {'enabled' if opts.daemon else 'disabled - exits after this run'}")


def run(settings: Settings, opts: Options, log_file: Path) -> int:
    areas = Areas(make_storage(settings, CHAT), make_storage(settings, TEMP_VIDEOS))
    agent = ChatAgent(settings, EventTools(settings, telegram=None))  # notify() sends nothing itself
    queued = read_inbox(areas.chat)

    def next_batch(held: set[str]) -> list[Event]:
        due = [e for e in read_inbox(areas.chat) if e.inbox_key not in held]
        return due[:opts.limit] if opts.limit else due

    if opts.daemon:
        print_header(agent, areas, opts, len(next_batch(set())), len(queued), log_file)
        return run_daemon(log, "events", next_batch, lambda batch: process_batch(agent, areas, batch, opts))
    batch = next_batch(set())
    print_header(agent, areas, opts, len(batch), len(queued), log_file)
    return 1 if process_batch(agent, areas, batch, opts) else 0


def process_batch(agent: ChatAgent, areas: Areas, batch: list[Event], opts: Options) -> list[str]:
    """Process `batch`; returns the inbox keys of the events that failed."""
    t_run = time.perf_counter()
    sent = skipped = 0
    failed: list[str] = []
    for i, event in enumerate(batch, 1):
        summary = (event.body.get("summary") or {}).get("description") or "(no description)"
        log.info(f"[{i}/{len(batch)}] Processing event {event.event_id} "
                 f"({PurePosixPath(event.inbox_key).name})")
        log.info(f"  Recorded at: {event.body.get('recorded_at')}; description: {summary}")
        try:
            if process_event(agent, areas, event, opts):
                sent += 1
            else:
                skipped += 1
        except Exception as e:  # noqa: BLE001
            failed.append(event.inbox_key)
            log.exception(f"  Event processing failed: {e} - inbox file retained for the next run")

    remaining = sum(1 for _ in read_inbox(areas.chat))
    log.info(f"Run completed in {time.perf_counter() - t_run:.1f}s: {sent} message(s) queued, "
             f"{skipped} suppressed, {len(failed)} failed; {remaining} remaining in inbox")
    return failed


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="Override EVENT_PROCESS_LIMIT (events per run, 0 = no limit)")
    parser.add_argument("--preserve", action="store_true",
                        help="Testing: keep the inbox files (events stay queued) and the LLM request/response")
    parser.add_argument("--daemon", action="store_true",
                        help="Keep running: poll the inbox every second while it is empty (--limit = per batch)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Also show the other modules' internal INFO logs")
    args = parser.parse_args(argv)
    if args.daemon and args.preserve:
        parser.error("--daemon cannot be combined with --preserve (the same events would be processed forever)")
    log_file = setup_logging(PROGRAM, args.verbose, log.name)
    settings = load_settings()
    return run(settings, Options(
        limit=settings.event_process_limit if args.limit is None else args.limit,
        preserve=args.preserve,
        daemon=args.daemon,
    ), log_file)


if __name__ == "__main__":
    raise SystemExit(main())
