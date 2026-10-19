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
    # The active day's notes are already in the system prompt: this is only for other days.
    _tool("get_day_notes",
          "Read the daily-context file for a specific date OTHER than the active day. The active "
          "day's notes are already injected into the system prompt under `## Day notes`; only call "
          "this for a different past date the user asks about or wants to correct.",
          {"date": _DATE}, ("date",)),
]

# Tools that change long-term memory or the day notes: only the memory model
# runs them. Only chat (the user's instructions) has them; notify() has no tools.
MEMORY_TOOLS = {"remember", "forget", "note_today", "clear_today"}
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
    "fulfills": (list[str], Field(
        default_factory=list,
        description="When you send a message: the text of each `alert-on:` request in the day notes that "
                    "this message fulfils, copied as written (without the prefix). The request is then "
                    "closed. Empty when you skip, or for `alert-each:` requests.")),
}
