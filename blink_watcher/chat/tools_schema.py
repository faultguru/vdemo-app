"""The chat agent's tools as the model sees them, and its answer schema.

The LLM request has no native tool calls: each round the model answers with
an AgentStep — the tool calls to run and/or the final reply — whose schema
(step_schema) limits the tool names to the role's tools. The declarations
below (name, description, JSON-schema parameters) are listed in the system
prompt (chat.prompts.tools_block); chat.agent runs the calls.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model


def _tool(name: str, description: str, properties: Optional[dict] = None,
          required: tuple[str, ...] = ()) -> dict:
    params: dict = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = list(required)
    return {"name": name, "description": description, "parameters": params}


_DATE = {"type": "string", "description": "Date in YYYY-MM-DD format."}

TOOLS: list[dict] = [
    _tool("list_event_dates", "List all dates that have recorded events, as YYYY-MM-DD strings."),
    _tool("search_events",
          "Search events across several days by words in their description, people, labels, "
          "threat assessment, camera and notification (any word matches; best matches first). "
          "Use it for questions beyond one day, e.g. whether a person, vehicle or activity was "
          "seen before. Dates default to the conversation's lookback period.",
          {"text": {"type": "string", "description": "Words to look for, e.g. traits of a person."},
           "start_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
           "end_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
           "camera": {"type": "string", "description": "Only this camera."}},
          ("text",)),
    _tool("view_event_pictures",
          "Look at the pictures (thumbnails) of up to 6 events yourself — they are attached to "
          "your next request, the user does not see them. Use it to compare people or vehicles "
          "visually, e.g. with a picture the user sent.",
          {"event_ids": {"type": "array", "items": {"type": "string"}}},
          ("event_ids",)),
    _tool("list_events", "List compact summaries of events recorded on a given date.",
          {"date": _DATE}, ("date",)),
    _tool("get_event_details", "Return the full JSON record for a specific event_id.",
          {"event_id": {"type": "string"}, "date": {"type": "string", "description": "Optional YYYY-MM-DD hint."}},
          ("event_id",)),
    _tool("send_thumbnail", "Send the thumbnail image for an event to the user's Telegram chat.",
          {"event_id": {"type": "string"}, "date": {"type": "string"}}, ("event_id",)),
    _tool("send_video", "Send the event's video (the processed clip, trimmed to the activity) to the user's chat.",
          {"event_id": {"type": "string"}, "date": {"type": "string"}}, ("event_id",)),
    _tool("remember",
          "Save a durable fact to long-term memory: user preferences, trusted people/vehicles, "
          "schedules, what to ignore, etc. Use for facts that should survive across conversations.",
          {"note": {"type": "string", "description": "One concise sentence stating the fact to remember."}},
          ("note",)),
    _tool("forget",
          "Remove memory entries whose text contains the given substring. "
          "Use when the user says 'forget X' or corrects a prior fact.",
          {"substring": {"type": "string", "description": "Text to match against stored memory entries."}},
          ("substring",)),
    _tool("note_today",
          "Append a short line to the daily context file — only when the user tells you something "
          "to keep for the day: facts that may not hold tomorrow (outfits, who a person in a clip was, "
          "matched by appearance) or a one-day request (an expected delivery to be told about). "
          "Generic, stable facts go to `remember` instead. The target day is chosen automatically "
          "— you do NOT pick a date.",
          {"text": {"type": "string", "description": "One concise line describing the fact."},
           "time": {"type": "string",
                    "description": "Optional HH:MM (24-hour). Pass the event's own time from the describer "
                                   "output when the note refers to a specific event, so the file stays "
                                   "in chronological order."}},
          ("text",)),
    _tool("clear_today",
          "Remove lines from the daily context whose text contains the given substring. Use when a "
          "tracked situation resolves (package picked up, visitor left) or was misread. The target "
          "day matches the most recently looked-at event; you do NOT pick a date.",
          {"substring": {"type": "string", "description": "Text to match against context entries."}},
          ("substring",)),
    _tool("note_for_date",
          "Append a line to the day notes of a day still to come (today or later), for something "
          "the user expects on that day: a delivery, a visitor, a worker, an alert request for that "
          "day. The line applies on that day like any day note (alert-on: / alert-each: included). "
          "Resolve the user's day (tomorrow, a weekday, a date) against the current date.",
          {"date": _DATE,
           "text": {"type": "string", "description": "One concise line, in the day-note formats."},
           "time": {"type": "string", "description": "Optional HH:MM (24-hour) the note is about."}},
          ("date", "text")),
    _tool("clear_for_date",
          "Remove lines containing the given substring from the day notes of a coming day "
          "(e.g. an expected delivery that was cancelled).",
          {"date": _DATE,
           "substring": {"type": "string", "description": "Text to match against that day's entries."}},
          ("date", "substring")),
    _tool("set_security_mode",
          "Set the security mode now, when the user says they are leaving, back, at home or going "
          "away. Give `until` when they say when it ends (when they will be back): the mode then turns "
          "back to normal on its own at that time.",
          {"mode": {"type": "string", "enum": ["normal", "home", "away", "vacation"]},
           "until": {"type": "string", "description": "Optional local time YYYY-MM-DDTHH:MM it ends."},
           "note": {"type": "string", "description": "Optional short reason (e.g. at work)."}},
          ("mode",)),
    _tool("schedule_security_mode",
          "Schedule a security mode for a later period (e.g. a trip): it is set on its own at `start` "
          "and turns back to normal at `end`. Scheduled changes are listed under the security mode in "
          "the system prompt; periods must not overlap.",
          {"mode": {"type": "string", "enum": ["normal", "home", "away", "vacation"]},
           "start": {"type": "string", "description": "Local time YYYY-MM-DDTHH:MM it starts."},
           "end": {"type": "string", "description": "Local time YYYY-MM-DDTHH:MM it ends; omit for until changed."},
           "note": {"type": "string", "description": "Optional short reason (e.g. family trip)."}},
          ("mode", "start")),
    _tool("cancel_security_schedule",
          "Cancel a scheduled security mode by the id shown in the system prompt's scheduled changes.",
          {"id": {"type": "string"}}, ("id",)),
    # The active day's notes are already in the system prompt: this is only for other days.
    _tool("get_day_notes",
          "Read the daily-context file for a specific date OTHER than the active day. The active "
          "day's notes are already injected into the system prompt under `## Day notes`; only call "
          "this for a different past date the user asks about or wants to correct.",
          {"date": _DATE}, ("date",)),
]

# Tools that change long-term memory or the day notes: only the memory model
# runs them. Only chat (the user's instructions) has them; notify() has no tools.
MEMORY_TOOLS = {"remember", "forget", "note_today", "clear_today", "note_for_date", "clear_for_date"}
# Tools that change the security mode: also run by the memory model (they act on
# the user's plans); the web console's agent has them too.
MODE_TOOLS = {"set_security_mode", "schedule_security_mode", "cancel_security_schedule"}
STATE_TOOLS = MEMORY_TOOLS | MODE_TOOLS
# The web console's agent can't change memory and has no day notes.
CONSOLE_TOOLS = [t for t in TOOLS if t["name"] not in MEMORY_TOOLS | {"get_day_notes"}]


# ---------- the answer format ----------

class ToolCall(BaseModel):
    name: str = Field(description="Name of the tool to call.")
    args_json: str = Field(description='The arguments as a JSON object, e.g. {"date": "2026-10-17"}; "{}" for none.')


class AgentStep(BaseModel):
    tool_calls: list[ToolCall] = Field(description="Tools to run now, in order; empty when only replying.")
    reply: str = Field(description="The final reply; empty while waiting for tool results.")


def step_schema(tools: list[dict], extra: Optional[dict] = None) -> type[AgentStep]:
    """AgentStep whose tool names are limited to the role's tools — the
    answer schema itself shows which tools exist (without native tools a
    model may otherwise conclude it has none). `extra`: more answer fields
    for the role, as create_model field definitions."""
    names = tuple(t["name"] for t in tools)
    if not names:
        return create_model(
            "AgentStep", __base__=AgentStep,
            tool_calls=(list[ToolCall], Field(default_factory=list, max_length=0,
                                              description="Always empty: no tools in this role.")),
            **(extra or {}),
        )
    call = create_model(
        "ToolCall", __base__=ToolCall,
        name=(Literal[names], Field(description="The tool to call: " + ", ".join(names) + ".")),
    )
    return create_model(
        "AgentStep", __base__=AgentStep,
        tool_calls=(list[call], Field(description="Tools to run now, in order (all of them are available "
                                                  "in this session); empty when only replying.")),
        **(extra or {}),
    )


# notify(): besides the reply, which earlier events show the same subject —
# stitches a person / vehicle from camera to camera, sent or not.
NOTIFY_FIELDS = {
    "same_subject_as": (list[str], Field(
        default_factory=list,
        description="event_ids of the day's earlier events (any camera) that show the same person(s) or "
                    "vehicle as this event; empty if none. Fill it whether or not you send a message.")),
    "subject": (str, Field(
        default="",
        description="Short trait signature of the main subject (who / what, clothing, build, carried "
                    "objects, vehicle), as seen in this event; empty if there is no person or vehicle.")),
    "household_vehicle": (Literal["departed", "arrived", "none"], Field(
        default="none",
        description="Only when the input says auto-away is on: 'departed' if this event shows a household "
                    "vehicle (as long-term memory describes it) leaving the property, 'arrived' if it shows "
                    "one coming home (see the system prompt for the evidence that counts); 'none' otherwise, "
                    "and whenever unsure. Fill it whether or not you send a message.")),
    "fulfills": (list[str], Field(
        default_factory=list,
        description="When you send a message: the text of each `alert-on:` request in the day notes that "
                    "this message fulfils, copied as written (without the prefix). The request is then "
                    "closed. Empty when you skip, or for `alert-each:` requests.")),
}
