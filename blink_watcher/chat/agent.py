"""The chat agent — one agent, two roles, on the configured LLM provider
(blink_watcher.ai, LLM_PROVIDER):

  handle(chat_id, text)       a chat message from the user -> the reply (chat bot, web console)
  notify(event, request_dir)  an incoming event -> a message or SKIP (pipeline.notify)

Both run the same tool loop (ChatAgent._run). Rounds go to the fast model
(CHAT_AGENT_MODEL / CHAT_AGENT_EFFORT); when it asks for a memory change
(MEMORY_TOOLS: long-term memory or day notes), that round is redone by the
memory model (CHAT_AGENT_MEMORY_MODEL / CHAT_AGENT_MEMORY_EFFORT), which then
finishes the turn. Memory changes only come from the user's chat
instructions: notify() has no tools — it reads memory, day notes and the
day's earlier events, and only answers with the message.

The model answers each round with an AgentStep (chat.tools_schema): tool
calls to run and/or the final reply; tool results are fed back in the next
round, up to MAX_TOOL_ROUNDS. The prompts are assembled in chat.prompts.
Every round's request goes to <request_dir>/round-<n>[-memory]/ (the
provider's request files + agent.step.json, the parsed answer) and, in full,
to STORAGE_ROOT/logs/chat_prompts.log.

Per-chat history is a sliding window of the last text turns, persisted per
day in the chat area (chat.store.ChatHistory); console mode (web console)
keeps it in memory only, reads long-term memory without changing it and has
no day notes. Tool-call round trips happen inside one turn and are not
persisted.
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional

from blink_watcher.ai import ImagePart, LLMRequest, Part, TextPart, Usage, make_llm
from blink_watcher.chat import prompts
from blink_watcher.chat import store as chat_store
from blink_watcher.chat.memory import DayNotes, Memory
from blink_watcher.chat.tools import EventToolError, EventTools
from blink_watcher.chat.tools_schema import (
    CONSOLE_TOOLS, MEMORY_TOOLS, NOTIFY_FIELDS, TOOLS, AgentStep, step_schema,
)
from blink_watcher.core import security_mode
from blink_watcher.core.runtime import logs_dir
from blink_watcher.pipeline.events import notification_prompt
from blink_watcher.settings import Settings

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6
MAX_PICTURES = 6           # view_event_pictures: events per call
DEFAULT_LOOKBACK_DAYS = 14
PROMPT_LOG = "chat_prompts.log"

Progress = Callable[..., None]


def _prompt_logger(settings: Settings) -> logging.Logger:
    """The full requests, to STORAGE_ROOT/logs/chat_prompts.log only — never
    to the console. Not a child of this module's logger: dictConfig turns
    propagation back on for children of configured loggers."""
    logger = logging.getLogger("blink_watcher_prompts")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(logs_dir(settings) / PROMPT_LOG, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        logger.addHandler(handler)
    return logger


def _preview(value, limit: int = 160) -> str:
    """One line, at most `limit` characters, for the log."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


# ---------- results ----------

@dataclass
class Turn:
    """One run of the tool loop: the final reply and what it took, summed over the rounds."""
    provider: str
    reply: str = ""
    models: list[str] = field(default_factory=list)     # the model of each round, in order
    rounds: int = 0
    duration_ms: int = 0
    retries: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0                                # at API prices, where the provider reports it
    stop_reason: Optional[str] = None                    # of the last round
    tool_calls: list[dict] = field(default_factory=list)  # {name, args, result}
    answer: dict = field(default_factory=dict)            # the final answer's role-specific fields

    def record(self) -> dict:
        """As stored in the notifier's result file."""
        return {"model": self.models[-1] if self.models else None, "models": self.models,
                "backend": self.provider, "rounds": self.rounds, "duration_ms": self.duration_ms,
                "retries": self.retries, "usage": vars(self.usage).copy(),
                "cost_usd_equivalent": self.cost_usd, "stop_reason": self.stop_reason,
                "tool_calls": self.tool_calls, "answer": self.answer}


TONES = ("cautious", "friendly", "informative")


@dataclass
class Notification:
    """The notify role's final reply, parsed (SKIP: <reason> / <tone> text)."""
    send: bool
    tone: Optional[str]
    message: str
    skip_reason: str
    reply: str
    models: list[str] = field(default_factory=list)           # models that answered, in order
    same_subject_as: list[str] = field(default_factory=list)  # earlier events with the same subject
    subject: str = ""                                          # trait signature of the main subject


def parse_notification(reply: str) -> Notification:
    text = reply.strip()
    if not text:
        return Notification(False, None, "", "empty reply", reply)
    if text.upper().startswith("SKIP"):
        return Notification(False, None, "", text[4:].lstrip(" :-").strip() or "no reason given", reply)
    tone = None
    for t in TONES:
        if text.lower().startswith(f"<{t}>"):
            tone, text = t, text[len(t) + 2:].lstrip()
            break
    return Notification(True, tone, text, "", reply)


# ---------- the agent ----------

class ChatAgent:
    """One agent, two roles — handle() for the user's chat messages and
    notify() for incoming events — on the same tool loop (_run)."""

    def __init__(self, settings: Settings, tools: EventTools, console: bool = False):
        """console (web console): history kept in memory only, long-term
        memory read-only, no day notes (neither shown nor written)."""
        self.settings = settings
        self.tools = tools
        self.console = console
        self.llm = make_llm(settings)
        self.model = settings.chat_agent_model                # fast model: every round starts here
        self.effort = settings.chat_agent_effort or None
        self.memory_model = settings.chat_agent_memory_model  # memory changes
        self.memory_effort = settings.chat_agent_memory_effort or None
        self.history = (chat_store.MemoryHistory() if console
                        else chat_store.ChatHistory(settings.chat_dir, settings.timezone))
        self.memory = Memory(settings.chat_dir)
        self.day_notes = None if console else DayNotes(settings.chat_dir, settings.timezone)
        self.prompt_log = _prompt_logger(settings)
        self._history_lock = asyncio.Lock()
        self._active_day: Optional[str] = None  # notify(): the event's day; handle(): the focus date
        self._lookback: Optional[tuple[str, str]] = None  # handle(): default search_events range

    @property
    def tool_list(self) -> list[dict]:
        """The chat role's tools; the console's agent can't change memory or use day notes."""
        return CONSOLE_TOOLS if self.console else TOOLS

    async def clear_history(self, chat_id: str) -> int:
        """Start today's history window empty. Returns prior length."""
        async with self._history_lock:
            prior = self.history.clear(chat_id)
        log.info(f"  Chat history cleared: {prior} turn(s) removed from today's window")
        return prior

    def _day(self) -> Optional[str]:
        """The active day, else the most recent date with events — where
        `note_today` / `clear_today` write in chat. None if there are no events."""
        if self._active_day:
            return self._active_day
        dates = self.tools.list_event_dates()
        return dates[-1] if dates else None

    # ---------- tools ----------

    async def _invoke_tool(self, name: str, args: dict) -> dict:
        try:
            if name == "list_event_dates":
                return {"dates": self.tools.list_event_dates()}
            if name == "list_events":
                return {"events": self.tools.list_events(args["date"])}
            if name == "search_events":
                start, end = args.get("start_date"), args.get("end_date")
                if not (start and end) and self._lookback:
                    start, end = start or self._lookback[0], end or self._lookback[1]
                return {"matches": self.tools.search_events(args["text"], start, end, args.get("camera"))}
            if name == "view_event_pictures":
                paths = self.tools.event_pictures(list(args.get("event_ids") or [])[:MAX_PICTURES])
                return {"status": f"{len(paths)} picture(s) attached to your next request",
                        "pictures": [{"event_id": i, "path": p} for i, p in paths]}
            if name == "get_event_details":
                return {"event": self.tools.get_event_details(args["event_id"], args.get("date"))}
            if name == "send_thumbnail":
                return {"status": await self.tools.send_thumbnail(args["event_id"], args.get("date"))}
            if name == "send_video":
                return {"status": await self.tools.send_video(args["event_id"], args.get("date"))}
            if name == "remember":
                return {"status": self.memory.remember(args["note"])}
            if name == "forget":
                return {"status": self.memory.forget(args["substring"])}
            if name == "note_today":
                return {"status": self.day_notes.note(args["text"], date=self._day(), time=args.get("time"))}
            if name == "clear_today":
                return {"status": self.day_notes.clear(args["substring"], date=self._day())}
            if name == "get_day_notes":
                return {"date": args["date"], "notes": self.day_notes.read(args["date"]) or "(empty)"}
            return {"error": f"unknown tool: {name}"}
        except EventToolError as e:
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return {"error": f"{type(e).__name__}: {e}"}

    # ---------- the loop ----------

    async def _run(self, base: str, tools: list[dict], messages: list[dict], request_dir: Path,
                   progress: Progress, events_block: Optional[str] = None, extra: Optional[dict] = None,
                   attachments: Optional[list[Part]] = None) -> Turn:
        """Ask for an AgentStep, run its tool calls, feed the results back,
        until the model gives the final reply (at most MAX_TOOL_ROUNDS rounds).
        A reply given together with tool calls is final unless a tool failed.
        `extra`: role-specific answer fields, returned in Turn.answer."""
        system = prompts.system_prompt(base, self.memory, self.day_notes, self.settings.timezone, self._day(),
                                       security_mode=security_mode.prompt_block(security_mode.current(self.settings)))
        if events_block:
            system += "\n\n" + events_block
        system += "\n\n" + (prompts.tools_block(tools) if tools else prompts.no_tools_block())
        allowed = {t["name"] for t in tools}
        schema = step_schema(tools, extra)
        turn = Turn(self.llm.name)
        tool_log: list[str] = []
        memory_mode = False  # once the memory model took over, it answers the rest of the turn
        # Sent with every round after the transcript: the user's pictures
        # (attachments) + event pictures the agent asked to look at.
        images: list[Part] = list(attachments or [])

        async def ask(round_idx: int, user_text: str, model: str, effort: Optional[str], tag: str) -> AgentStep:
            round_dir = request_dir / f"round-{round_idx}{tag}"
            progress(f"[round {round_idx}] Sending request to {model} ({self.llm.name}, effort {effort or 'default'})"
                     + (f" with {len(tool_log)} tool result(s)" if tool_log else "") + "...")
            response = await self.llm.ask_async(
                LLMRequest(system, [TextPart(user_text), *images], schema, model, effort), round_dir, progress)
            turn.rounds += 1
            turn.models.append(model)
            turn.duration_ms += response.duration_ms
            turn.retries += response.retries
            turn.usage.add(response.usage)
            turn.cost_usd += response.cost_usd or 0
            turn.stop_reason = response.stop_reason
            step = response.parsed
            if step is None:
                progress(f"[round {round_idx}] Response received without a usable answer", logging.ERROR)
                raise RuntimeError(f"no structured answer from {model}: {response.raw_text!r}")
            (round_dir / "agent.step.json").write_text(step.model_dump_json(indent=2), encoding="utf-8")
            answer = ([f"{len(step.tool_calls)} tool call(s): " + ", ".join(c.name for c in step.tool_calls)]
                      if step.tool_calls else []) + (["final reply"] if step.reply.strip() else [])
            progress(f"[round {round_idx}] Response received in {response.duration_ms / 1000:.1f}s "
                     f"({response.usage.input_tokens} input / {response.usage.output_tokens} output tokens): "
                     + (" + ".join(answer) or "empty answer"))
            return step

        for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
            user_text = prompts.transcript(messages, tool_log)
            self.prompt_log.info("LLM request (round %d):\n%s\n\n%s", round_idx, system, user_text)
            if memory_mode:
                step = await ask(round_idx, user_text, self.memory_model, self.memory_effort, "-memory")
            else:
                step = await ask(round_idx, user_text, self.model, self.effort, "")
                wanted = sorted({c.name for c in step.tool_calls} & MEMORY_TOOLS)
                if wanted and self.memory_model != self.model:
                    progress(f"[round {round_idx}] Memory change requested ({', '.join(wanted)}) - round handed "
                             f"over to {self.memory_model}; {self.model}'s answer discarded")
                    memory_mode = True
                    step = await ask(round_idx, user_text, self.memory_model, self.memory_effort, "-memory")

            failed = False
            for call in step.tool_calls:
                try:
                    args = json.loads(call.args_json or "{}")
                except json.JSONDecodeError as e:
                    args, result = {}, {"error": f"args_json is not valid JSON: {e}"}
                else:
                    progress(f"[round {round_idx}] Running tool {call.name} {_preview(args)}...")
                    result = (await self._invoke_tool(call.name, args) if call.name in allowed
                              else {"error": f"tool not available here: {call.name}"})
                failed = failed or "error" in result
                for pic in result.get("pictures") or []:
                    images += [TextPart(f"Picture of event {pic['event_id']}:"), self.tools.picture(pic["path"])]
                turn.tool_calls.append({"name": call.name, "args": args, "result": result})
                if "error" in result:
                    progress(f"[round {round_idx}] Tool {call.name} failed: {_preview(result['error'])}",
                             logging.WARNING)
                else:
                    progress(f"[round {round_idx}] Tool {call.name} completed: {_preview(result)}")
                tool_log.append(f"{len(tool_log) + 1}. {call.name} {json.dumps(args, ensure_ascii=False)} -> "
                                f"{json.dumps(result, ensure_ascii=False)}")
            turn.answer = {k: getattr(step, k) for k in (extra or {})}
            if (step.reply.strip() and not failed) or not step.tool_calls:
                turn.reply = step.reply.strip()
                return turn
            if step.reply.strip():
                progress(f"[round {round_idx}] Reply discarded - a tool failed, the results are sent back")
        progress(f"Tool loop stopped after {MAX_TOOL_ROUNDS} rounds without a final reply", logging.WARNING)
        turn.reply = "(tool loop exceeded)"
        return turn

    # ---------- roles ----------

    async def notify(self, event: dict, request_dir: Path,
                     progress: Optional[Progress] = None) -> tuple[Notification, Turn, str]:
        """Incoming event (pipeline.notify): with the long-term memory, the
        event day's notes and its earlier events (with their notification
        decisions), decide whether the event is worth a message — what the
        user asked to hear about or not, earlier alerts, routine activity —
        and write it. No tools: memory and day notes are only read. Stateless
        w.r.t. chat history (the message enters it once sent). Returns
        (notification, turn, per-event prompt)."""
        progress = progress or (lambda msg, level=logging.INFO: log.log(level, msg))
        prompt = notification_prompt(event.get("recorded_at") or "", event.get("summary") or {},
                                     chat_store.should_offer_media(self.settings.chat_dir), event.get("camera"),
                                     event.get("security_mode"))
        self._active_day = chat_store.event_date(event["event_id"])
        try:
            earlier = prompts.events_block(self.tools, self._active_day, exclude=event["event_id"],
                                           hint="the current event is not in this list.")
            turn = await self._run(prompts.NOTIFY, [], [{"role": "user", "text": prompt}], request_dir, progress,
                                   events_block=earlier, extra=NOTIFY_FIELDS)
        finally:
            self._active_day = None
        notification = parse_notification(turn.reply)
        notification.models = turn.models
        notification.same_subject_as = list(turn.answer.get("same_subject_as") or [])
        notification.subject = (turn.answer.get("subject") or "").strip()
        return notification, turn, prompt

    async def handle(self, chat_id: str, user_text: str, focus_date: Optional[str] = None,
                     lookback_days: Optional[int] = None, attachments: Optional[list[Part]] = None,
                     context: str = "", context_images: Optional[list[Part]] = None) -> str:
        """A user's chat message (chat bot, web console) -> the reply ("" =
        send nothing, e.g. after a picture or video was sent).

        focus_date (web console): the day the user is looking at — questions
        are about it unless they clearly are not; it is preloaded and day
        notes go to it. lookback_days (web console): default period for
        questions beyond that day, or for all questions without a focus date
        (search_events default range). attachments: the user's pictures (and
        their labels) sent with the message. context / context_images (live
        monitor): system prompt text and parts (live snapshots) for this
        message only — not kept in the history."""
        console = prompts.CONSOLE_NOTE if self.console else ""
        if not (focus_date or lookback_days):
            return await self._handle(chat_id, user_text, console + context, attachments, context_images)
        self._active_day = focus_date
        today = chat_store.local_now(self.settings.timezone).date()
        days = lookback_days or DEFAULT_LOOKBACK_DAYS
        self._lookback = ((today - timedelta(days=days - 1)).isoformat(), today.isoformat())
        try:
            return await self._handle(chat_id, user_text,
                                      prompts.scope_block(focus_date, self._lookback, days) + console + context,
                                      attachments, context_images)
        finally:
            self._active_day, self._lookback = None, None

    async def _handle(self, chat_id: str, user_text: str, scope: str, attachments: Optional[list[Part]],
                      context_images: Optional[list[Part]]) -> str:
        def progress(msg: str, level: int = logging.INFO) -> None:
            log.log(level, f"  {msg.strip()}")

        async with self._history_lock:
            history = self.history.load(chat_id)
            pictures = sum(isinstance(a, ImagePart) for a in attachments or [])
            history.append({"role": "user", "text": user_text + (f"\n[sent {pictures} picture(s)]" if pictures else "")})
            self.history.save(chat_id, history)
        progress(f"Chat history loaded: {len(history) - 1} earlier turn(s); message added"
                 + (" (in memory only)" if self.console else
                    f" ({self.history.path(chat_id).relative_to(self.settings.chat_dir).as_posix()})"))
        progress(f"Generating reply with {self.model} (memory changes: {self.memory_model}) - "
                 f"day notes {self._day() or '(no events yet)'}, {len(self.tool_list)} tools available...")

        with tempfile.TemporaryDirectory(prefix="chat-agent-") as tmp:
            turn = await self._run(prompts.CHAT + scope, self.tool_list, history, Path(tmp), progress,
                                   events_block=prompts.events_block(self.tools, self._day()),
                                   attachments=[*(attachments or []), *(context_images or [])])
        cost = f", ~${turn.cost_usd:.4f}" if turn.cost_usd else ""
        progress(f"Reply generated in {turn.rounds} round(s), {turn.duration_ms / 1000:.1f}s, "
                 f"{len(turn.tool_calls)} tool call(s), models {' > '.join(dict.fromkeys(turn.models))}{cost}")

        reply = turn.reply
        silent = not reply or reply.upper() == "SILENT"  # SILENT = media already sent, no follow-up text
        # The history records what went out: media sent by tools ("[sent video
        # event_id=...]") and the text reply, if any.
        text = "\n".join(t for t in (chat_store.media_turn_text(turn.tool_calls), "" if silent else reply) if t)
        if text:
            async with self._history_lock:
                fresh = self.history.load(chat_id)
                fresh.append({"role": "model", "text": text})
                self.history.save(chat_id, fresh)
            progress(f"Chat history updated: {_preview(text)}")
        if silent:
            progress("No follow-up message: " + ("media already sent (SILENT)" if reply else "empty reply"))
            return ""
        return reply
