"""Gemini-powered chat agent with event-tool function calling.

Per-chat history is a sliding window of the last 10 user/model text turns,
persisted as JSON under ``settings.chat_history_dir``. Tool-call
round-trips happen inside a single user turn and are not persisted.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .config import Settings
from .daily import DailyContext
from .event_tools import EventTools, EventToolError
from .memory import Memory

log = logging.getLogger(__name__)

HISTORY_WINDOW = 10
MAX_TOOL_ROUNDS = 6

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
LOGS_DIR = PROJECT_ROOT / "logs"


def _build_prompt_logger() -> logging.Logger:
    """Dedicated logger for full Gemini request dumps.

    Writes to ``logs/chat_prompts.log`` and does NOT propagate to the
    root logger, so prompt bodies never reach the console.
    """
    logger = logging.getLogger("blink_watcher.chat_agent.prompts")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "_bw_prompt_handler", False)
        for h in logger.handlers
    ):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(
            LOGS_DIR / "chat_prompts.log", encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
        )
        handler._bw_prompt_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


prompt_log = _build_prompt_logger()




def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not ln.lstrip().startswith("#")]
    return "\n".join(kept).strip()


SYSTEM_INSTRUCTION = _load_prompt("chat.system.txt")
NOTIFY_SYSTEM_INSTRUCTION = _load_prompt("chat_notify.system.txt")


def _daily_write_tool_declarations() -> list[dict]:
    return [
        {
            "name": "note_today",
            "description": (
                "Append a short line to the daily context file. This is the "
                "PREFERRED place to store facts — prefer it over `remember` "
                "for almost everything, including identifications, corrections "
                "(\"that person was me\"), clothing, schedules, vehicles, "
                "what someone is doing, who is inside, packages, etc. Also "
                "use after a notification is delivered to record the event "
                "signature so a near-identical recurrence can be suppressed. "
                "The target day is chosen automatically from whichever event "
                "you most recently looked at — you do NOT pick a date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "One concise line describing the fact.",
                    },
                    "time": {
                        "type": "string",
                        "description": (
                            "Optional HH:MM (24-hour). Pass the event's own "
                            "time from the describer output when the note "
                            "refers to a specific event, so the file stays "
                            "in chronological order."
                        ),
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "clear_today",
            "description": (
                "Remove lines from the daily context whose text contains the "
                "given substring. Use when a tracked situation resolves "
                "(package picked up, visitor left) or was misread. The "
                "target day matches the most recently looked-at event; you "
                "do NOT pick a date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "substring": {
                        "type": "string",
                        "description": "Text to match against context entries.",
                    },
                },
                "required": ["substring"],
            },
        },
    ]


def _daily_chat_tool_declarations() -> list[dict]:
    """Chat-only daily tools: writes plus a read for non-active days.

    The active day's notes are already injected into the system prompt,
    so `get_day_notes` is only needed when the user asks about an older
    day that is not the latest event day.
    """
    return _daily_write_tool_declarations() + [
        {
            "name": "get_day_notes",
            "description": (
                "Read the daily-context file for a specific date OTHER "
                "than the active day. The active day's notes are already "
                "injected into the system prompt under `## Day notes`; "
                "only call this for a different past date the user asks "
                "about or wants to correct."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format.",
                    },
                },
                "required": ["date"],
            },
        },
    ]


def _tool_declarations() -> list[dict]:
    return [
        {
            "name": "list_event_dates",
            "description": "List all dates that have recorded events, as YYYY-MM-DD strings.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "list_events",
            "description": "List compact summaries of events recorded on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                },
                "required": ["date"],
            },
        },
        {
            "name": "get_event_details",
            "description": "Return the full JSON record for a specific event_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "date": {"type": "string", "description": "Optional YYYY-MM-DD hint."},
                },
                "required": ["event_id"],
            },
        },
        {
            "name": "send_thumbnail",
            "description": "Send the thumbnail image for an event to the user's Telegram chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["event_id"],
            },
        },
        {
            "name": "send_video",
            "description": "Send the full video for an event to the user's Telegram chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["event_id"],
            },
        },
        {
            "name": "remember",
            "description": (
                "Save a durable fact to long-term memory: user preferences, "
                "trusted people/vehicles, schedules, what to ignore, etc. "
                "Use for facts that should survive across conversations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "One concise sentence stating the fact to remember.",
                    },
                },
                "required": ["note"],
            },
        },
        {
            "name": "forget",
            "description": (
                "Remove memory entries whose text contains the given substring. "
                "Use when the user says 'forget X' or corrects a prior fact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "substring": {
                        "type": "string",
                        "description": "Text to match against stored memory entries.",
                    },
                },
                "required": ["substring"],
            },
        },
        *_daily_chat_tool_declarations(),
    ]


class ChatAgent:
    def __init__(self, settings: Settings, tools: EventTools):
        if not settings.gemini_api_key:
            raise SystemExit("GEMINI_API_KEY must be set in .env")
        if not settings.gemini_chat_model:
            raise SystemExit("GEMINI_CHAT_MODEL must be set in .env")
        self.settings = settings
        self.tools = tools
        self.model = settings.gemini_chat_model
        self.history_dir = settings.chat_history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.memory = Memory(settings.state_dir / "memory.md")
        try:
            tz = ZoneInfo(settings.timezone)
        except Exception:  # noqa: BLE001
            tz = None
        self.daily = DailyContext(settings.state_dir / "daily", tz)

        from google import genai
        from google.genai import types as genai_types
        self._genai = genai
        self._types = genai_types
        self._client = genai.Client(api_key=settings.gemini_api_key)

        self._tool_config = self._types.Tool(
            function_declarations=_tool_declarations()
        )
        self._notify_tool_config = self._types.Tool(
            function_declarations=_daily_write_tool_declarations()
        )
        self._history_lock = asyncio.Lock()

    # ---------- history persistence ----------

    def _history_path(self, chat_id: str) -> Path:
        return self.history_dir / f"{chat_id}.json"

    def _load_history(self, chat_id: str) -> list[dict]:
        p = self._history_path(chat_id)
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.warning("corrupt history for %s, resetting", chat_id)
            return []

    def _save_history(self, chat_id: str, history: list[dict]) -> None:
        trimmed = history[-HISTORY_WINDOW:]
        self._history_path(chat_id).write_text(
            json.dumps(trimmed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def clear_history(self, chat_id: str) -> int:
        """Delete the persisted history file for a chat. Returns prior length."""
        async with self._history_lock:
            p = self._history_path(chat_id)
            prior = 0
            if p.exists():
                try:
                    prior = len(json.loads(p.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001
                    prior = 0
                p.unlink(missing_ok=True)
        log.info("history cleared for %s (%d turns)", chat_id, prior)
        return prior

    def _to_contents(self, history: list[dict]) -> list[Any]:
        t = self._types
        return [
            t.Content(role=h["role"], parts=[t.Part.from_text(text=h["text"])])
            for h in history
        ]

    # ---------- memory-aware system prompt ----------

    def _system_with_memory(self, base: str) -> str:
        body = self.memory.read_body()
        if not body:
            body = "(empty)"
        try:
            tz = ZoneInfo(self.settings.timezone)
        except Exception:  # noqa: BLE001
            tz = None
        now = datetime.now(tz) if tz else datetime.now().astimezone()
        now_block = (
            f"## Current date\n"
            f"Today is {now.strftime('%A, %Y-%m-%d')} "
            f"(local time {now.strftime('%H:%M %Z')}).\n"
            f"- \"today\" = {now.strftime('%Y-%m-%d')}\n"
            f"- \"yesterday\" = {(now - timedelta(days=1)).strftime('%Y-%m-%d')}\n"
            f"Always resolve relative dates (today, yesterday, last night, "
            f"this morning, an hour ago, N days ago) against the above. "
            f"Never use a date earlier than this for \"today\".\n"
        )
        active_day = self._latest_event_date() or now.strftime("%Y-%m-%d")
        daily_body = self.daily.read_body(date=active_day)
        if not daily_body:
            daily_body = "(empty)"
        return (
            f"{base}\n\n{now_block}\n"
            f"## Current memory\n{body}\n\n"
            f"## Day notes ({active_day})\n"
            f"These are the daily-context entries for the latest event day "
            f"on disk. Any `note_today` / `clear_today` calls you make also "
            f"target this same day automatically.\n"
            f"{daily_body}\n"
        )

    # ---------- prompt logging ----------

    def _content_to_dict(self, c: Any) -> dict:
        parts_out: list[dict] = []
        for p in getattr(c, "parts", None) or []:
            if getattr(p, "text", None):
                parts_out.append({"text": p.text})
            elif getattr(p, "function_call", None) is not None:
                fc = p.function_call
                parts_out.append({
                    "function_call": {"name": fc.name, "args": dict(fc.args or {})}
                })
            elif getattr(p, "function_response", None) is not None:
                fr = p.function_response
                parts_out.append({
                    "function_response": {
                        "name": fr.name,
                        "response": dict(fr.response or {}),
                    }
                })
            else:
                parts_out.append({"unknown": repr(p)})
        return {"role": getattr(c, "role", None), "parts": parts_out}

    def _log_prompt(self, round_idx: int, contents: list[Any]) -> None:
        payload = {
            "model": self.model,
            "system_instruction": SYSTEM_INSTRUCTION,
            "contents": [self._content_to_dict(c) for c in contents],
        }
        prompt_log.info(
            "gemini request (round %d):\n%s",
            round_idx,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    # ---------- tool dispatch ----------

    def _latest_event_date(self) -> str | None:
        """Return the most recent date for which events.*.json exists.

        This is the single source of truth for where `note_today` and
        `clear_today` write: whichever day the camera system most
        recently recorded events on, regardless of what the LLM has
        been looking at in this turn. Returns None if no event files
        exist at all.
        """
        dates = self.tools.list_event_dates()
        return dates[-1] if dates else None

    async def _invoke_tool(self, name: str, args: dict) -> dict:
        try:
            if name == "list_event_dates":
                return {"dates": self.tools.list_event_dates()}
            if name == "list_events":
                return {"events": self.tools.list_events(args["date"])}
            if name == "get_event_details":
                return {"event": self.tools.get_event_details(
                    args["event_id"], args.get("date")
                )}
            if name == "send_thumbnail":
                msg = await self.tools.send_thumbnail(
                    args["event_id"], args.get("date")
                )
                return {"status": msg}
            if name == "send_video":
                msg = await self.tools.send_video(
                    args["event_id"], args.get("date")
                )
                return {"status": msg}
            if name == "remember":
                return {"status": self.memory.remember(args["note"])}
            if name == "forget":
                return {"status": self.memory.forget(args["substring"])}
            if name == "note_today":
                date = self._latest_event_date()
                return {"status": self.daily.note(
                    args["text"],
                    date=date,
                    time=args.get("time"),
                )}
            if name == "clear_today":
                date = self._latest_event_date()
                return {"status": self.daily.clear(
                    args["substring"], date=date
                )}
            if name == "get_day_notes":
                body = self.daily.read_body(date=args["date"])
                return {"date": args["date"], "notes": body or "(empty)"}
            return {"error": f"unknown tool: {name}"}
        except EventToolError as e:
            return {"error": str(e)}
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return {"error": f"{type(e).__name__}: {e}"}

    # ---------- main entrypoint ----------

    async def append_notification(
        self, chat_id: str, text: str, event_id: str | None = None
    ) -> None:
        """Inject a server-initiated notification into chat history."""
        tag = f"[auto-notification event_id={event_id}]" if event_id else "[auto-notification]"
        async with self._history_lock:
            history = self._load_history(chat_id)
            history.append({"role": "model", "text": f"{tag} {text}"})
            self._save_history(chat_id, history)

    async def compose_notification(self, prompt: str) -> str:
        """Ask Gemini to rewrite a raw event description as a user-facing alert.

        Stateless w.r.t. chat history, but the composer can call
        `note_today` / `clear_today` tools as a side effect to track
        state across events (e.g. package-drop → package-pickup). The
        returned text is the final alert (or SKIP) after any tool calls.

        Daily writes always go to the latest event date on disk, picked
        up automatically by `_invoke_tool` via `_latest_event_date`.
        """
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=self._system_with_memory(NOTIFY_SYSTEM_INSTRUCTION),
            tools=[self._notify_tool_config],
            temperature=0.4,
        )
        contents: list[Any] = [
            t.Content(role="user", parts=[t.Part.from_text(text=prompt)])
        ]
        log.info("notify compose prompt: %s", prompt)

        final_text: str = ""
        for round_idx in range(MAX_TOOL_ROUNDS):
            self._log_prompt(round_idx, contents)
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self.model,
                contents=contents,
                config=config,
            )
            candidate = (response.candidates or [None])[0]
            if candidate is None or candidate.content is None:
                final_text = getattr(response, "text", "") or ""
                break

            model_content = candidate.content
            contents.append(model_content)

            fn_parts = [
                p for p in (model_content.parts or [])
                if getattr(p, "function_call", None) is not None
            ]
            if not fn_parts:
                final_text = getattr(response, "text", "") or ""
                break

            response_parts = []
            for p in fn_parts:
                fc = p.function_call
                args = dict(fc.args or {})
                log.info("notify tool call: %s(%s)", fc.name, args)
                result = await self._invoke_tool(fc.name, args)
                response_parts.append(
                    t.Part.from_function_response(name=fc.name, response=result)
                )
            contents.append(t.Content(role="user", parts=response_parts))
        else:
            final_text = "(tool loop exceeded)"

        text = final_text.strip()
        if not text:
            raise RuntimeError("compose_notification returned empty text")
        log.info("notify compose result: %s", text)
        return text

    async def handle(self, chat_id: str, user_text: str) -> str:
        async with self._history_lock:
            history = self._load_history(chat_id)
            history.append({"role": "user", "text": user_text})
            self._save_history(chat_id, history)

        t = self._types
        contents: list[Any] = self._to_contents(history)
        config = t.GenerateContentConfig(
            system_instruction=self._system_with_memory(SYSTEM_INSTRUCTION),
            tools=[self._tool_config],
            temperature=0.3,
        )

        final_text: str = ""
        for round_idx in range(MAX_TOOL_ROUNDS):
            self._log_prompt(round_idx, contents)
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self.model,
                contents=contents,
                config=config,
            )
            candidate = (response.candidates or [None])[0]
            if candidate is None or candidate.content is None:
                final_text = getattr(response, "text", "") or ""
                break

            model_content = candidate.content
            contents.append(model_content)

            fn_parts = [
                p for p in (model_content.parts or [])
                if getattr(p, "function_call", None) is not None
            ]
            if not fn_parts:
                final_text = getattr(response, "text", "") or ""
                break

            response_parts = []
            for p in fn_parts:
                fc = p.function_call
                args = dict(fc.args or {})
                log.info("tool call: %s(%s)", fc.name, args)
                result = await self._invoke_tool(fc.name, args)
                response_parts.append(
                    t.Part.from_function_response(name=fc.name, response=result)
                )
            contents.append(t.Content(role="user", parts=response_parts))
        else:
            final_text = "(tool loop exceeded)"

        final_text = final_text.strip()
        if not final_text or final_text.upper() == "SILENT":
            # Empty model text OR the SILENT sentinel after a successful media
            # send: suppress the follow-up chat message and do NOT persist
            # anything in history either.
            return ""
        async with self._history_lock:
            fresh = self._load_history(chat_id)
            fresh.append({"role": "model", "text": final_text})
            self._save_history(chat_id, fresh)
        return final_text
