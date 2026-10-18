"""The chat agent's prompts: the role instructions (prompts/chat*.txt,
agent_*.txt) and the sections assembled around them.

System prompt = the role's instructions (+ the console's scope / live monitor
notes) + current date, security mode, long-term memory and day notes (system_prompt) + the
active day's events (events_block) + the tool protocol with the role's tools
(tools_block) or the no-tools answer format. The user message is the
conversation transcript with this turn's tool results (transcript).
"""
from __future__ import annotations

import json
from datetime import timedelta
from typing import Optional

from blink_watcher.chat.memory import DayNotes, Memory
from blink_watcher.chat.store import local_now
from blink_watcher.chat.tools import EventToolError, EventTools
from blink_watcher.core.prompts import load_prompt

CHAT = load_prompt("chat.system.txt")                     # the user's chat (Telegram, web console)
NOTIFY = load_prompt("chat_notify.system.txt")            # incoming events
CONSOLE = load_prompt("chat_console.txt")                 # web console chat
SCOPE_DAY = load_prompt("chat_scope_day.txt")             # web console: a day selected
SCOPE_LOOKBACK = load_prompt("chat_scope_lookback.txt")   # web console: no day selected
LIVE_MONITOR = load_prompt("chat_live.txt")               # web console: live monitor open
TOOL_PROTOCOL = load_prompt("agent_tools.txt")            # roles with tools
NO_TOOLS_PROTOCOL = load_prompt("agent_no_tools.txt")     # roles without tools (notify)

CONSOLE_NOTE = "\n\n## This chat\n\n" + CONSOLE


def scope_block(focus_date: Optional[str], lookback: tuple[str, str], days: int) -> str:
    """Which dates the user's questions are about (web console)."""
    period = f"the last {days} days ({lookback[0]} to {lookback[1]})"
    text = (SCOPE_DAY.format(focus_date=focus_date, period=period) if focus_date
            else SCOPE_LOOKBACK.format(period=period))
    return "\n\n## Scope of the conversation\n\n" + text


def live_block(cameras: str, now: str) -> str:
    """The live monitor the user is watching (web console): `cameras` a line
    per camera with its snapshot times, `now` the local time HH:MM:SS."""
    return "\n\n## Live monitor\n\n" + LIVE_MONITOR.format(cameras=cameras, now=now)


def system_prompt(base: str, memory: Memory, day_notes: Optional[DayNotes], tz_name: str,
                  active_day: Optional[str] = None, security_mode: str = "") -> str:
    """`base` + current date, the security mode now (core.security_mode.prompt_block),
    long-term memory and the day notes of `active_day` (default: today; no day
    notes if `day_notes` is None)."""
    body = memory.read() or "(empty)"
    now = local_now(tz_name)
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
    if security_mode:
        now_block += f"\n## Security mode now\n{security_mode}\n"
    if day_notes is None:
        return f"{base}\n\n{now_block}\n## Current memory\n{body}\n"
    active_day = active_day or now.strftime("%Y-%m-%d")
    notes = day_notes.read(active_day) or "(empty)"
    return (
        f"{base}\n\n{now_block}\n"
        f"## Current memory\n{body}\n\n"
        f"## Day notes ({active_day})\n"
        f"These are the daily-context entries for the latest event day "
        f"on disk. Any `note_today` / `clear_today` calls you make also "
        f"target this same day automatically.\n"
        f"{notes}\n"
    )


def events_block(tools: EventTools, day: Optional[str], exclude: Optional[str] = None,
                 hint: str = "call get_event_details only when you need more detail (appearance, objects, "
                             "timings), list_events for other days.") -> str:
    """The active day's events (without `exclude`), preloaded so most
    questions need no tool round."""
    if not day:
        return "## Events (none recorded yet)"
    try:
        events = [e for e in tools.list_events(day) if e.get("event_id") != exclude]
    except EventToolError:
        events = []
    lines = [f"## Events on {day} ({len(events)}, oldest first)",
             "Preloaded with the notification decision of each; " + hint]
    lines += [json.dumps(e, ensure_ascii=False) for e in events]
    return "\n".join(lines)


def tools_block(tools: list[dict]) -> str:
    """The tool protocol + one line per tool: name(args): description."""
    lines = [TOOL_PROTOCOL, ""]
    for t in tools:
        params = t.get("parameters") or {}
        required = set(params.get("required") or [])
        args = ", ".join(
            f"{name}{'' if name in required else '?'}: {spec.get('description') or spec.get('type', '')}".rstrip(": ")
            for name, spec in (params.get("properties") or {}).items()
        )
        lines.append(f"- {t['name']}({args}): {' '.join(t['description'].split())}")
    return "\n".join(lines)


def no_tools_block() -> str:
    return "## Answer format\n" + NO_TOOLS_PROTOCOL


def transcript(messages: list[dict], tool_log: list[str]) -> str:
    """The conversation (and this turn's tool results) as one user message."""
    parts = ["## Conversation"]
    for m in messages:
        parts.append(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['text']}")
    if tool_log:
        parts += ["", "## Tool calls made this turn", *tool_log, "",
                  "Continue: call more tools or give the final reply."]
    return "\n".join(parts)
