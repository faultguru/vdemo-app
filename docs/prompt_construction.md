# Prompt Construction

How `ChatAgent` assembles the system prompt it sends to Gemini on every
call — and why the pieces are arranged the way they are.

← [Back to overview](README.md)

## Sources on disk

All prompts are plain text files under `prompts/`, hand-editable, with
lines starting with `#` stripped at load time so comments never reach
the model.

| File | Used by |
| --- | --- |
| `prompts/chat.system.txt` | `ChatAgent.handle` — interactive chat |
| `prompts/chat_notify.system.txt` | `ChatAgent.compose_notification` — proactive alerts |
| `prompts/describe_video.system.txt` | `describe_video.py` — video describer (see [video_analysis.md](video_analysis.md)) |
| `prompts/describe_video.claude.user.txt` | Claude video describer user-turn |
| `prompts/describe_video.gemini.user.txt` | Gemini video describer user-turn |

Loading happens once at module import time via `_load_prompt`:

```
SYSTEM_INSTRUCTION = _load_prompt("chat.system.txt")
NOTIFY_SYSTEM_INSTRUCTION = _load_prompt("chat_notify.system.txt")
```

So a prompt edit requires a process restart to take effect.

## `_system_with_memory(base)`

Both `handle` and `compose_notification` wrap the base prompt with
live state before passing it to Gemini. The wrapper is defined in
`chat_agent.py` and produces a string with this shape:

```
<base prompt from chat.system.txt or chat_notify.system.txt>

## Current date
Today is <day name>, <YYYY-MM-DD> (local time <HH:MM TZ>).
- "today" = <YYYY-MM-DD>
- "yesterday" = <YYYY-MM-DD>
Always resolve relative dates (today, yesterday, last night,
this morning, an hour ago, N days ago) against the above. Never
use a date earlier than this for "today".

## Current memory
<full contents of state_dir/memory.md, header stripped>

## Day notes (<active_day>)
These are the daily-context entries for the latest event day on
disk. Any `note_today` / `clear_today` calls you make also target
this same day automatically.
<contents of state_dir/daily/<active_day>.md, last 30 lines>
```

The **active day** is computed as:

```
self._latest_event_date() or now.strftime("%Y-%m-%d")
```

`_latest_event_date()` returns `EventTools.list_event_dates()[-1]`,
i.e. the newest `events.YYYY-MM-DD.json` file on disk. Fall-back to
wall-clock only if the events directory is completely empty.

This is the *same* day that `note_today` and `clear_today` write to
(see [memory.md](memory.md)), so the LLM's read view and write view
are always aligned. No tool call is needed for the active day's
notes — they're already in the prompt.

## Why pre-fetch vs tool call

Earlier iterations had `get_day_notes` as a tool the LLM would call
to read the daily file. That pattern has two costs:

1. **Extra LLM round-trip.** The notifier would produce a tool call,
   the backend would return the notes, then the notifier would run
   again with the result. Two Gemini calls per event instead of one.
2. **Unreliable context.** The LLM might forget to call the read
   tool, or call it with the wrong date, and produce an alert
   without the duplicate-check information.

Pre-fetching into the system prompt sidesteps both. The notifier is
now allowed only *write* daily tools; reads are baked into the
prompt.

## Tool sets — two flavors

`chat_agent.py` defines three factories:

- `_daily_write_tool_declarations()` — `note_today`, `clear_today`.
- `_daily_chat_tool_declarations()` — the writes above plus
  `get_day_notes` for reading *other* past dates on user request.
- `_tool_declarations()` — the full chat set: event read/send tools
  + `remember`/`forget` + `_daily_chat_tool_declarations()`.

The two Tool configs:

```
self._tool_config         = Tool(function_declarations=_tool_declarations())
self._notify_tool_config  = Tool(function_declarations=_daily_write_tool_declarations())
```

So the notifier sees a *strict subset* of what the chat agent sees:
only the two daily writes, no event lookups, no long-term memory
mutation, no media sends. The notifier's job is
"read-only-for-events, write-to-daily-only" and the tool schema
enforces that shape.

## Section ordering in `chat.system.txt`

The base prompt is structured so the LLM reads, in order:

1. Role statement.
2. The list of tools it has available (informational — the Gemini
   function_declarations are the authoritative schema, but the prose
   list helps the model reason about when to use each).
3. Date-resolution rule (relative dates resolve against
   `## Current date` injected below).
4. Media-send behavior (reply with `SILENT` after successful send).
5. **Memory section** — "prefer daily over persistent; persistent is
   identity-level only" plus the canonical `identified:` and
   `alert-on:` formats.
6. User-facing phrasing rule ("today" only means the real calendar
   day, not the day of a video under discussion).

Injection of `## Current date` / `## Current memory` / `## Day notes`
happens after all of this, so the runtime state appears below the
behavioral rules.

## Section ordering in `chat_notify.system.txt`

The notifier prompt is structured around a decision pipeline:

1. Role — home-security assistant composing Telegram alerts.
2. Tone selection — `<cautious>` / `<friendly>` / `<informative>`.
3. Rules — tone tag, time phrasing, `offer_media` field, no
   greetings, output format.
4. **Long-term memory section** — HARD OVERRIDE (persistent opt-ins
   that beat every SKIP rule), general SKIP discretion.
5. **HARD OVERRIDE (today only)** — `alert-on:` daily opt-ins that
   also beat every SKIP rule.
6. **Day notes correlation protocol** — step-by-step decision tree:
   - Step 0: Identification lookup. Scan `identified:` lines, match
     to subject, resolve to household → SKIP.
   - Step 1: Context correlation (package drops, visitors inside).
   - Step 2: Duplicate-suppression via `alerted:` signatures.
7. `note_today` / `clear_today` usage — when to record, when to
   clear, what canonical format to use.
8. Hero case — package pickup is HIGH URGENCY.

The identification lookup is explicitly labeled "RUN THIS FIRST"
and pre-empts all other suppression so an `identified: ... =
household` line reliably kills the next similar event.

## Prompt logging

Every call to `compose_notification` and `handle` logs the full
request payload to `logs/chat_prompts.log` via a dedicated
`logging.Logger` named `blink_watcher.chat_agent.prompts`. That
logger has `propagate = False` and a single `FileHandler` installed
at module load time, so prompt bodies never reach the console.

The payload is JSON with three keys:

```json
{
  "model": "gemini-3-flash-preview",
  "system_instruction": "<full chat.system.txt text>",
  "contents": [
    {"role": "user", "parts": [{"text": "..."}]},
    {"role": "model", "parts": [{"function_call": {...}}]},
    ...
  ]
}
```

`logs/chat_prompts.log` grows unboundedly — rotate or truncate it
manually as needed.

## See also

- [chatbots.md](chatbots.md) — how the two paths actually invoke
  Gemini and process responses.
- [memory.md](memory.md) — what the `## Current memory` and
  `## Day notes` sections contain and how they're written.
