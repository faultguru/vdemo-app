# blink-watcher

A home-security pipeline that listens to Blink cloud motion events,
downloads each clip, runs it through an LLM video describer
(Claude or Gemini), writes structured event records to disk, and
exposes a Telegram chat agent that can answer questions about what
happened, send pictures and videos, and suppress duplicate alerts
using a lightweight daily-memory system.

**End-to-end shape:** Blink → SQLite clip queue → describer → event
JSON → notifier LLM → Telegram. All state lives on the local disk;
the only external services are Blink, your chosen describer provider,
and the Telegram Bot API.

See [`docs/README.md`](docs/README.md) for the full architecture
overview, sub-docs, and worked examples.

## Install

```bash
pip install -e .
```

Python 3.11+ required. YOLOv8 weights (`yolov8s.pt`) are expected in
`models/` — on first run Ultralytics will download them automatically
if the file isn't already there.

## Configuration

All runtime knobs live in `.env`; nothing sensitive is hardcoded in
source. Copy the example and fill in your values:

```bash
cp .env.example .env
# edit .env and set BLINK_USERNAME / BLINK_PASSWORD / GEMINI_API_KEY / ...
```

`.env` is already in `.gitignore` — never commit it. The full list
of supported variables (with inline comments) lives in
[`.env.example`](.env.example).

Claude users: authentication piggybacks on the OAuth token from
`~/.claude/.credentials.json` — run `claude` once to sign in. No API
key goes in `.env` for Claude.

## Commands

All commands are installed as console scripts by `pip install -e .`.

### `blink-watcher` — live poller + downloader

```bash
blink-watcher run                              # live poll + download loop
blink-watcher backfill --hours 24              # backfill last 24h
blink-watcher backfill --days 3 --then-live    # backfill then transition to live
blink-watcher backfill --hours 6 --dry-run     # count clips without downloading
blink-watcher backfill --hours 6 --restart     # ignore saved cursor
blink-watcher telegram "hello"                 # send a one-shot Telegram message
blink-watcher telegram "*bold*" --markdown
```

### `blink-describe` — run the video describer on a clip

```bash
blink-describe path/to/clip.mp4
```

Writes structured analysis JSON alongside the clip. See
[`docs/video_analysis.md`](docs/video_analysis.md) for the pipeline
(OpenCV triage → motion segmentation → YOLOv8 → LLM) and the output
schema.

### `blink-event` — event-log utilities

```bash
blink-event --help
```

Appends and queries records in `events/events.<DATE>.json`. See
[`docs/event_loading.md`](docs/event_loading.md) for how an event
travels from the camera to a notifier-ready outbox file.

### `blink-chat` — Telegram notifier + chat agent

```bash
blink-chat
# equivalent to: python -m blink_watcher.chat_bot
```

Runs two concurrent loops in one process:

- **Notifier** — drains `events/outbox/*.json` and decides whether
  each event deserves an alert. Produces a tone-tagged Telegram
  message (`<cautious>` / `<friendly>` / `<informative>`) or emits
  `SKIP: <reason>` to suppress.
- **Chat agent** — long-polls Telegram for messages from the
  authorized `TELEGRAM_CHAT_ID` and answers them with
  `GEMINI_CHAT_MODEL`.

See [`docs/chatbots.md`](docs/chatbots.md) for the full wiring.

**Chat-agent function-calling tools:**

| Tool | Purpose |
| --- | --- |
| `list_event_dates` | Dates with recorded events |
| `list_events(date)` | Compact summaries for one date |
| `get_event_details(event_id)` | Full event JSON |
| `send_thumbnail(event_id)` | Send the event thumbnail image |
| `send_video(event_id)` | Send the full event video |
| `remember` / `forget` | Identity-level persistent memory |
| `note_today` / `clear_today` | Daily-scratchpad memory |
| `get_day_notes(date)` | Read an older day's notes |

**Sandbox:** all file access is restricted to `./events/`. Paths
escaping via `..` or absolute paths are rejected.

**Context:** per-chat sliding window of the last 10 user/model turns,
persisted at `data/chat_history/<chat_id>.json`. Auto-notifications
the notifier sent while the user was away are replayed into history
so the chat agent knows what the user has already been told.

**Memory:** two stores — persistent facts in `data/memory.md` (edited
via `remember` / `forget`) and per-day scratchpad in
`data/daily/YYYY-MM-DD.md` (edited via `note_today` / `clear_today`).
See [`docs/memory.md`](docs/memory.md).

**Example conversation:**

```
you: what happened today?
bot: (list_event_dates → list_events) Two events — delivery drop-off
     at 9:40 PM and a pickup at 9:51 PM.
you: send me the thumbnail of the second one
bot: (send_thumbnail) [photo delivered] SILENT
you: that was my neighbour, let me know if you see them again
bot: (note_today) OK, I'll alert you on any further sightings today.
```

## Layout

```
blink_watcher/
  main.py              # blink-watcher CLI (run / backfill / telegram)
  poller.py            # live event polling
  backfill.py          # historical backfill
  store.py             # SQLite clip queue (WAL mode)
  downloader.py        # concurrent clip downloads
  auth.py              # Blink login + session cache
  blink_api.py         # blinkpy wrapper
  describers.py        # Claude + Gemini video describers
  describe_video.py    # blink-describe CLI (motion pipeline)
  event_log.py         # blink-event CLI
  event_tools.py       # sandboxed event-query tools
  telegram.py          # Telegram Bot API client
  chat_bot.py          # blink-chat entrypoint (notify + long-poll loops)
  chat_agent.py        # dual-role Gemini agent (notifier + chat)
  memory.py            # persistent identity memory
  daily.py             # per-day ephemeral memory
  config.py            # pydantic-settings .env loader
prompts/               # hand-editable prompt files (#-comments stripped at load)
  chat.system.txt
  chat_notify.system.txt
  describe_video.system.txt
  describe_video.claude.user.txt
  describe_video.gemini.user.txt
models/                # YOLOv8 weights (gitignored)
events/                # events.<DATE>.json + per-day clips/thumbnails + outbox/
videos/                # raw downloaded clips
data/                  # store.sqlite, sessions, chat_history/, memory.md, daily/
logs/                  # chat_prompts.log + runtime logs
docs/                  # architecture docs (start with docs/README.md)
```

## Documentation

- [`docs/README.md`](docs/README.md) — architecture overview with
  high-level diagram and module map.
- [`docs/event_loading.md`](docs/event_loading.md) — how a motion
  event travels from Blink to the outbox.
- [`docs/video_analysis.md`](docs/video_analysis.md) — the describer
  pipeline and output schema.
- [`docs/chatbots.md`](docs/chatbots.md) — the notifier and chat
  agent, sharing one `ChatAgent` instance.
- [`docs/prompt_construction.md`](docs/prompt_construction.md) — how
  system prompts are assembled, where memory gets injected.
- [`docs/memory.md`](docs/memory.md) — persistent vs daily memory,
  canonical line formats, hand-editing.
- [`docs/examples.md`](docs/examples.md) — worked examples of every
  file shape the pipeline produces.

## Troubleshooting

- **`getUpdates` returns nothing:** a webhook may be set on the bot.
  Clear it: `curl "https://api.telegram.org/bot<TOKEN>/deleteWebhook"`.
- **Gemini 404:** `GEMINI_CHAT_MODEL` or `GEMINI_VIDEO_MODEL` is an
  invalid model id — swap to a valid one.
- **Telegram "file too large":** Telegram bots cap uploads at 50 MB
  for `sendVideo` / `sendDocument`.
- **`GEMINI_API_KEY must be set`:** the Gemini describer and chat
  agent both need it in `.env`; there is no default.
- **`CAMERAS must list at least one camera name`:** the `CAMERAS`
  env var is empty or missing. Comma-separate real camera names as
  they appear in the Blink app.
- **Claude auth fails:** run `claude` once to generate
  `~/.claude/.credentials.json`. The describer reads `accessToken`
  from that file — no API key needed.

## Prompts

All LLM prompts live in [`prompts/`](prompts/) as plain text. Lines
starting with `#` are stripped at load time and never sent to the
model, so you can leave editorial notes in-line. Prompt edits require
a process restart to take effect.
