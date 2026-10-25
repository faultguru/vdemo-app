# Faultguru Watchtower

Faultguru Watchtower (Python package `blink_watcher`) is a home-security
assistant for cloud cameras. It downloads every motion clip from the camera
provider (Blink today), throws away the ones where nothing
happens using plain computer vision (OpenCV motion, YOLO person check), has an
LLM describe the ones that matter from a handful of key frames, and lets a
chat agent decide which events are worth a Telegram message. You can talk to
the same agent on Telegram ("who was that?", "show me the video", "don't
tell me about the gardener today") and from a local web console that also
shows the day's events, groups related clips into activities and has a live
view of the cameras.

## How it works

```
 camera cloud (Blink)
        |  clips + the provider's own detections
        v
 [fetch]    download new clips -> videos.raw/   queue -> videos.processed/inbox/
        |
        v
 [process]  triage -> OpenCV motion -> YOLO person check      (no AI; most clips stop here)
            trim to the activity -> sample key frames -> LLM description (told the security mode)
            security mode HOME: indoor cameras' clips saved as they are, no analysis
            -> videos.processed/ (clip, thumbnail, .metadata.json)    queue -> chat/events-inbox/
        |
        v
 [notify]   chat agent: send or skip? (long-term memory, today's notes, earlier events)
            -> chat/<date>/events.json   message -> chat/outbox/
        |
        v
 [chat-bot] Telegram: sends the outbox, answers your messages (same chat agent + tools)

 [webui]    local console: events, activities (stitch), clips in the pipeline,
            live monitor, chat with the analyst, configuration page,
            security mode (NORMAL / HOME / AWAY / VACATION; also /home, /away... on Telegram)
```

Each box is a separate service (its own process). They only talk through
files in the storage areas, so each one can be stopped, restarted or rerun on
its own. `python -m blink_watcher run` starts them all.

The camera provider and the LLM provider are both behind small interfaces
(`blink_watcher.providers`, `blink_watcher.ai`): Blink and Claude (through the
`claude` CLI or the Messages API) are implemented; see
[docs/architecture.md](docs/architecture.md) for adding e.g. Ring or Gemini.

## Quick start

Requirements: Python 3.11+, the [`claude` CLI](https://docs.claude.com/en/docs/claude-code)
signed in (`claude` once) for the default LLM provider, a Blink account and a
Telegram bot (from @BotFather).

```bash
pip install -e .                    # extras: .[claude-api] .[gemini] .[directml]
cp .env.example .env                # then set at least BLINK_USERNAME, BLINK_PASSWORD, CAMERAS,
                                    # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
python -m blink_watcher fetch --list    # first Blink login (asks for the 2FA PIN Blink e-mails:
                                        # rerun with BLINK_2FA_CODE=<pin>)
python -m blink_watcher run         # every service; Ctrl+C stops them all
```

`run.bat` (double-click on Windows) / `run.ps1` / `run.sh` do the same from the
project folder (using `.venv` when there is one). The console is at http://127.0.0.1:9090; its configuration page
(`/settings`) edits `.env` with every option documented, and shows the
cameras as a table: each camera's Enabled switch and Download filter
(Always / Person / Person or vehicle).

To run it in the background instead, use `vdemo` (`vdemo.bat` on Windows, from
the project folder; `vdemo help` shows the usage):

```bash
vdemo start      # every service in the background, no window; returns once the console answers
vdemo stop       # stops them (also a run in a console window)
vdemo restart
vdemo status     # exit code 0 = running, 3 = not running
vdemo console    # in this console, like run.bat (--only / --skip / -v)
```

Only one instance runs at a time. `GET /api/health` answers 200 when every
module is OK and 503 otherwise, for outside monitoring. For the Faultguru app
monitor, copy `docs/faultguru.sample.json` to `.faultguru` in the project folder
and set `root` (`.faultguru` is not committed).

## Commands

`python -m blink_watcher <command> [options]` (or the `blink-watcher` script);
`<command> --help` shows the options.

| command     | what it does |
|-------------|--------------|
| `run`       | every service below in one console (`--only webui`, `--skip chat-bot`); restarts a service that crashes |
| `control`   | `start` / `stop` / `restart` / `status` of `run` in the background (what `vdemo` calls) |
| `fetch`     | download clips (`--daemon`: keep polling; `--date`, `--camera`, `--list`, `--limit`) |
| `process`   | analyze and describe queued clips (`--daemon`; `--limit`, `--preserve`, `--redo` for testing) |
| `notify`    | the chat agent's send / skip decision for each new event (`--daemon`) |
| `chat-bot`  | the Telegram bot: sends notifications, answers messages |
| `webui`     | the local web console (`--host`, `--port`) |
| `stitch`    | print a day's activities and why clips were linked (tune `STITCH_*`) |
| `describe`  | run the LLM describer on one clip's sampled frames |
| `frames`    | sample (and trim) a clip's frames the way the describer does, no LLM |
| `hwmonitor` | log GPU / CPU / RAM load every few seconds, crash-safe (diagnostics) |

The services can also be run one per terminal, e.g. `python -m blink_watcher
process --daemon`. Without `--daemon`, `fetch`, `process` and `notify` do one
batch and exit.

## Configuration

Everything is configured in `.env` at the project root (environment variables
override it). [`.env.example`](.env.example) documents every option, grouped,
with the services that need a restart after a change; the web console's
configuration page is generated from it. Secrets (Blink password, Telegram
token) are only ever written to `.env`, never shown back.

The Claude providers use your Claude Code login — no API key in `.env`.

## Where files live

Everything the services write goes under `STORAGE_ROOT` (default
`./data-storage`):

| folder              | contents |
|---------------------|----------|
| `videos.raw/`       | clips as downloaded + their `.metadata.json`, `<date>/<Camera_Name>/<stamp>.mp4` |
| `videos.processed/` | kept clips (trimmed), thumbnails, full analysis metadata, `.event.json`; `inbox/` = queue of the video processor |
| `videos.ignored/`   | `.metadata.json` of clips that were dropped, with who dropped them and why |
| `videos.temp/`      | per-clip work folders while a clip is processed (frames, LLM requests) |
| `videos.live/`      | live monitor snapshots (kept `WEBUI_LIVE_KEEP_HOURS`) |
| `chat/`             | long-term memory, day notes, each day's `events.json`, chat history, `events-inbox/`, `outbox/` |
| `state/`            | camera provider session (`state/blink/session.json`), camera catalog (`state/cameras.json`) |
| `logs/`             | one rotating log per service (`fetch.log`, `process.log`, ..., `run.log`), `chat_prompts.log` |
| `models/`           | YOLO weights (downloaded on first use) |

Each storage area's folder and type are configurable (`<AREA>_PREFIX`,
`<AREA>_STORAGE_TYPE`); only local disk exists today. All of these files are
plain JSON / Markdown and safe to read; memory files are meant to be
hand-editable.

## Documentation

- [docs/architecture.md](docs/architecture.md) — packages, services, the camera and LLM provider interfaces (adding Ring, Gemini), settings, logging
- [docs/video_analysis.md](docs/video_analysis.md) — the video processor: check, trim, frame sampling, the describer
- [docs/chat.md](docs/chat.md) — the chat agent: notifications, Telegram chat, tools, memory and day notes, prompts
- [docs/webui.md](docs/webui.md) — the web console, its API, the live monitor, activities, the configuration page
- [docs/data_formats.md](docs/data_formats.md) — every file on disk and its format
