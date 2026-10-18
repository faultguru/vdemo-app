# The chat agent

One agent (`chat/agent.py`, `ChatAgent`) does three jobs on the configured
LLM provider:

| role | called by | in | out |
|------|-----------|----|-----|
| notify | `pipeline.notify` (the `notify` service) | one new event | a message to send, or skip |
| chat | `chat.bot` (the `chat-bot` service) | your Telegram message | the reply, pictures / videos of events |
| console | `webui` (the web console's chat) | a message from the browser | the reply, pictures / videos, in the browser |

All three see the same things: the long-term memory, the active day's notes
and the day's events. The difference is what they may do — notify has no
tools and never changes memory; chat has every tool; the console has the
event tools but cannot change memory or day notes.

## The tool loop

The LLM request has no native tool calls (the `claude -p` provider runs
without tools). Instead every round the model answers with an `AgentStep`
(`chat/tools_schema.py`): tool calls to run and/or the final reply, whose
schema limits the tool names to the role's tools. The agent runs the calls,
appends their results to the transcript and asks again, up to 6 rounds.

```
 system prompt = role instructions                    prompts/chat.system.txt | chat_notify.system.txt
                 (+ console scope / live monitor)     prompts/chat_console.txt, chat_scope_*.txt, chat_live.txt
               + current date, long-term memory, the active day's notes
               + the active day's events (compact)
               + tool protocol and the role's tools   prompts/agent_tools.txt | agent_no_tools.txt
 user message  = the conversation (history window) + this turn's tool calls and results
 answer        = AgentStep {tool_calls[], reply}      (+ notify: decision, tone, same_subject_as, subject)
```

Rounds go to the fast model (`CHAT_AGENT_MODEL`, `CHAT_AGENT_EFFORT`). When a
round asks for a memory change (`remember`, `forget`, `note_today`,
`clear_today`), that round is redone by the memory model
(`CHAT_AGENT_MEMORY_MODEL`, `CHAT_AGENT_MEMORY_EFFORT`), which finishes the
turn — the stronger model decides what is worth keeping. Every round's
request is written to `<request dir>/round-<n>[-memory]/` (the provider's
request files + `agent.step.json`) and in full to
`STORAGE_ROOT/logs/chat_prompts.log`.

Prompt assembly is in `chat/prompts.py`; the prompt texts are in `prompts/`
(they describe behaviour in general terms — the household's specifics live in
memory, not in prompts).

## Tools

`chat/tools.py` (`EventTools`) over the recorded events (`chat/<date>/events.json`):

| tool | |
|------|-|
| `list_event_dates`, `list_events`, `search_events`, `get_event_details` | find and read events (search: words in descriptions, people, labels, threats, camera; over a date range) |
| `view_event_pictures` | the model looks at up to 6 events' thumbnails itself (attached to its next round) |
| `send_thumbnail`, `send_video` | send an event's picture / trimmed clip to the user (Telegram, or the browser in the console) |
| `remember`, `forget` | long-term memory (chat role only) |
| `note_today`, `clear_today`, `get_day_notes` | day notes (chat role only) |

File paths in events are relative to STORAGE_ROOT and sandboxed: a path
resolving outside it is refused.

## Notifications (`notify`)

`python -m blink_watcher notify --daemon` takes each event from
`chat/events-inbox/` (oldest recording first):

1. **record** — the event goes into the day's list `chat/<date>/events.json`
   (replacing an entry with the same `event_id`).
2. **decide** — `ChatAgent.notify()` with `prompts/chat_notify.system.txt`.
   The per-event input (`pipeline.events.notification_prompt`) is the
   describer's summary, the time, the camera and whether media may be
   offered (at most once an hour). The model decides `send` or `skip` from
   what memory and today's notes say the user wants to hear about, earlier
   alerts on the day (no repeats for the same person doing the same thing)
   and routine activity, picks a tone (friendly / cautious / informative)
   and writes one or two sentences. It also names earlier events showing the
   same subject (`same_subject_as`), which become the event's `track` and
   link clips into activities in the console.
3. **queue** — the decision is stored with the event (`"notification"`);
   only for `send` a message is written to `chat/outbox/<stamp>-<CAMERA>.message.json`.
4. **clean up** — inbox file and the event's work folder removed (the
   request files of each round live in `videos.temp/<date>/<Camera>/<stamp>/event/`
   until then; `--preserve` keeps them).

## The Telegram bot (`chat-bot`)

`chat/bot.py` runs two loops on one Telegram client (`chat/telegram.py`):

- **outbox** — every 2 s, sends each message in `chat/outbox/` (text + the
  event's thumbnail as the photo), records it in the chat history and as
  `sent_at` in the event list, and removes the file.
- **updates** — long-polls Telegram. Only `TELEGRAM_CHAT_ID` is answered;
  other chats are dropped. `/clear` empties today's history window;
  `/normal`, `/home`, `/away`, `/vacation` set the security mode and `/mode`
  shows it (no LLM involved); any other message goes to `ChatAgent.handle()`
  with "typing..." shown meanwhile.

## Security mode

The security mode (`core/security_mode.py`: NORMAL / HOME / AWAY / VACATION) tells
the agents whether the residents are around. It is set from the console's
header or the Telegram commands above and kept in
`STORAGE_ROOT/state/security_mode.json`; HOME turns back to NORMAL at
`HOME_MODE_UNTIL` (default 23:00), AWAY and VACATION stay until changed.
Every chat and notification round gets the current mode in its system prompt
(`## Security mode now`, from `prompts/security_mode.txt`: the mode and what it
means), and each event carries the mode it was processed in (`security_mode`,
also in the per-event notification prompt), so the same activity can be
weighed differently — e.g. a person at an entrance while the residents are
away. The video describer gets the same block (see
[video_analysis.md](video_analysis.md)).

A network drop is logged in one line and retried. The chat history is a
sliding window of the last 10 text turns per chat, kept per day
(`chat/<date>/chat_history.<chat id>.json`); tool round trips inside a turn
are not kept.

## Memory and day notes

Two Markdown files the agent maintains from what you tell it — both meant to
be read and hand-edited (`chat/memory.py`):

- **Long-term memory**, `chat/memory/memory.md`: durable facts and
  preferences — who lives here and how to recognise them (by stable physical
  traits rather than clothing), trusted visitors and vehicles, what never to
  notify about. One `- (<date>) <fact>` bullet per entry. `remember` adds,
  `forget` removes by substring.
- **Day notes**, `chat/<date>/memory.md`: things that only hold for one day —
  "expecting a delivery", "ignore the garden today", who has been identified
  today, what was already alerted. One `[HH:MM] <text>` line per entry;
  `note_today` adds, `clear_today` removes by substring. Each day has its own
  file, so they roll over at midnight by themselves.

The chat prompt asks for two structured kinds of day note that the
notification prompt matches on:

- `identified: <trait signature> = <identity>` — when you say who someone
  was ("that's the gardener"); concrete traits (build, clothing, carried
  objects), with `(household)` appended for a household member, so later
  events of the same person are named — or skipped as routine.
- `alert-on: <subject / action / location>` — a one-day request ("tell me
  when the pizza arrives"); a matching event is notified even when other
  rules would skip it.

Both are in every system prompt (the active day's notes: the event's day for
notify, today for chat). Memory changes only come from your messages:
notifications read memory but never write it.

## The web console's chat

`webui` creates its own `ChatAgent(..., console=True)`: same prompts and
event tools, history kept in memory only (gone on restart), memory read but
never changed, no day notes. With a day selected in the console, questions
are about that day (beyond it: the last `WEBUI_CHAT_LOOKBACK_DAYS` days);
pictures and videos you attach are passed as images (a few frames of a video);
with the live monitor open, the agent also gets the latest snapshots of the
cameras on screen. Pictures and videos it sends come back to the browser.
See [webui.md](webui.md#chat).
