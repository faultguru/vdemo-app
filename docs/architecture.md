# Architecture

blink-watcher is five long-running services and a few tools, all in one
Python package (`blink_watcher`), started with `python -m blink_watcher
<command>`. The services never call each other: each one reads its input
queue from a storage area and writes its output to the next one. That keeps
the CPU-heavy video work (OpenCV, YOLO) away from the network loops, lets any
service be restarted or rerun by hand, and leaves every intermediate result
on disk to inspect.

```
 camera provider (Blink)
   |  clip list, clips (+ the provider's own detections)
   v
 fetch ------> videos.raw/<date>/<Camera>/          the clip + .metadata.json
   |           videos.processed/inbox/              queue
   v
 process ----> videos.processed/<date>/<Camera>/    kept: trimmed clip, thumbnail, metadata, .event.json
   |  LLM      videos.ignored/<date>/<Camera>/      dropped: metadata + why
   |           chat/events-inbox/                   queue
   v
 notify -----> chat/<date>/events.json              the event + the send / skip decision
   |  LLM      chat/outbox/                         a message to send
   v
 chat-bot <--> Telegram                             sends the outbox, answers your messages (LLM)

 webui         reads all of the above; live view through the camera provider;
               console chat through the LLM; edits .env (configuration page)
```

`python -m blink_watcher run` (`supervisor.py`) starts fetch, process,
notify, chat-bot and webui as child processes, prints their log lines in one
console and restarts one that exits (2 s, doubling up to 60 s while it keeps
failing). Ctrl+C stops them in turn (Windows: CTRL_BREAK to the child's own
process group, which `cli` turns into KeyboardInterrupt; POSIX: SIGINT) and
kills what has not stopped after 20 s.

## Packages

| package | role |
|---------|------|
| `cli.py`, `__main__.py` | `python -m blink_watcher <command>`: command -> module `main(argv)`, imported only when that command runs |
| `supervisor.py` | the `run` command (above) |
| `core/` | shared plumbing: `paths` (project folders, `.env`), `runtime` (`state/`, `logs/`, `models/` under STORAGE_ROOT), `storage` (storage areas, key layout), `logs` (`setup_logging`), `daemon` (inbox polling loop), `images` (`fit_jpeg`), `prompts` (`load_prompt`), `filters` (the per-camera detection filters: always / person / person_or_vehicle), `security_mode` (NORMAL / HOME / AWAY / VACATION: state, auto-end, prompt block) |
| `settings/` | `Settings`, composed of one section per area (see below); `document` = the configuration page's model of `.env.example` / `.env` |
| `providers/` | camera providers behind `CameraProvider` (`base`), `make_provider()`, the camera catalog (`catalog`); `blink/` = the Blink provider (blinkpy) |
| `ai/` | LLM providers behind `LLMClient` (`base`), `make_llm()`; `claude_cli/`, `claude_api/`; `gemini/` = a reference whole-video describer, not wired in |
| `vision/` | video analysis without AI: `triage`, `motion` (OpenCV), `yolo`, `check` (the three in order), `tracks`, `focus` (trim), `sampling` (key frames for the describer), `video`, `ffmpeg`, `hwtrace` |
| `describe/` | the video describer: sampled frames -> `VideoAnalysis` (`schema`) through `ai`; `request` builds the prompt, `cli` = the `describe` command |
| `pipeline/` | the queue services: `fetch`, `process/` (video processor), `notify`; `clip_metadata` (the clip record), `events` (event record, notification prompt) |
| `chat/` | the chat agent (`agent`), its `tools` and `tools_schema`, `prompts`, `memory` (long-term memory, day notes), `store` (chat area files), `telegram`, `bot` (the `chat-bot` service) |
| `live/` | the web console's live monitor: snapshot loop (`monitor`, `snapshots`) and MJPEG live view (`stream`) on the provider API |
| `stitch/` | groups a day's related clips into activities by fixed rules (`stitcher`, word lists in `vocab.json`) |
| `webui/` | the web console: `server` (app factory, `webui` command), `context`, `cameras` (the camera list), `incoming` (clips in the pipeline), `routes/` (one module per area of the page), `static/` (two single-page apps) |
| `tools/` | `hwmonitor` (diagnostics) |
| `prompts/` (project root) | every LLM prompt as a text file (`describe_*`, `chat*`, `agent_*`) |

Dependencies point one way: `core` and `settings` at the bottom; `providers`,
`ai`, `vision` on top of them; `describe` on `ai` + `vision`; `pipeline`,
`chat`, `live`, `stitch` on those; `webui` and `supervisor` at the top.
Nothing outside `providers/` imports blinkpy or `providers.blink`, and nothing
outside `ai/` builds a provider-specific request or calls `claude`.

## The services

| service | input | output | details |
|---------|-------|--------|---------|
| `fetch` | the provider's clip list (today, every `POLL_INTERVAL_SECONDS`) | `videos.raw/<date>/<Camera>/<stamp>.mp4` + `.metadata.json`; a copy of the metadata in `videos.processed/inbox/` | skips clips already stored; each camera's filter (`CAMERA_FILTERS`) on the provider's own detections; refreshes the camera catalog |
| `process` | `videos.processed/inbox/` | kept: `videos.processed/<date>/<Camera>/` (trimmed clip, thumbnail, metadata, `.event.json`) + `chat/events-inbox/`; dropped: `videos.ignored/` | [video_analysis.md](video_analysis.md) |
| `notify` | `chat/events-inbox/` | `chat/<date>/events.json` (event + decision), `chat/outbox/` (a message to send) | [chat.md](chat.md) |
| `chat-bot` | Telegram updates, `chat/outbox/` | Telegram messages, chat history | [chat.md](chat.md) |
| `webui` | everything above (read only) | live snapshots, `.env` (configuration page, from this PC only) | [webui.md](webui.md) |

`fetch`, `process` and `notify` do one batch and exit unless run with
`--daemon` (`core.daemon`: poll the inbox every second; a failed item waits
60 s before it is retried, and stays in the inbox until it succeeds).

## Camera providers

`providers/base.py` is everything the app needs from a camera cloud. A
provider is an async context manager holding one session:

```python
class CameraProvider(ABC):
    name: str                                                     # its CAMERA_PROVIDER value
    async def connect(self) -> None                               # AuthError / TwoFactorRequired / ProviderError
    async def close(self) -> None
    async def list_cameras(self) -> list[Camera]
    def list_clips(self, cameras, since, until) -> AsyncIterator[ProviderClip]
    def download_clip(self, clip) -> AsyncIterator[bytes]
    async def snapshots(self, cameras) -> dict[str, Snapshot]     # a new picture of each camera
    async def live_stream(self, camera) -> LiveSource             # ffmpeg input args (+ stdin feeding)
```

`Camera` (name, provider, id, model, has_own_person_detection) and
`ProviderClip` (provider_clip_id, camera, recorded_at in UTC, duration,
trigger mapped onto `motion` / `snapshot` / `live_view`, the provider's own
detections such as `person` / `vehicle`, raw record) are provider-neutral;
`pipeline.clip_metadata` turns a `ProviderClip` into the clip record that
every later step reads. Cameras are addressed by name; names match ignoring
case and surrounding spaces (`settings.cameras.camera_key`), as in `CAMERAS`.

The **camera catalog** (`providers/catalog.py`, `STORAGE_ROOT/state/cameras.json`)
is the account's camera list, saved whenever a provider session lists its
cameras (the fetch daemon at start and hourly, the live monitor on connect).
The web console and its configuration page read cameras only from it
(`load_cameras(settings)`), so they never need a provider session of their
own.

### Adding a provider (e.g. Ring)

1. `providers/ring/`: a `RingProvider(CameraProvider)` with `name = "ring"`,
   its login / session handling (keep any session file under
   `core.runtime.state_dir(settings, "ring")`), and a `RingSettings(BaseSettings)`
   section for its credentials.
2. Register it: `PROVIDERS["ring"] = "blink_watcher.providers.ring.provider:RingProvider"`
   in `providers/__init__.py`, add `"ring"` to `CameraSettings.camera_provider`'s
   `Literal`, and add `RingSettings` to `Settings`' bases in `settings/__init__.py`.
3. Document its keys in `.env.example` (a `# --- Ring account ---` group,
   `# @secret` on the password / token) so the configuration page shows them.
4. Map its clip triggers onto `MOTION` / `SNAPSHOT` / `LIVE_VIEW` and its object
   labels onto `person` / `vehicle` / ..., so the camera filters
   (`core.filters`, `CAMERA_FILTERS`) work unchanged. A live view that ffmpeg
   can read from a URL is a `UrlLiveSource(url)`; one the provider pushes
   as bytes is a `LiveSource` with `uses_stdin = True` (see `providers/blink/live.py`).

Nothing else changes: fetch, the live monitor and the web console only use
the interface.

## LLM providers

`ai/base.py` is one structured-output request and its normalised response:

```python
LLMRequest(system: str, parts: list[TextPart | ImagePart], schema: type[BaseModel],
           model: str, effort: str | None)
client = make_llm(settings)                       # LLM_PROVIDER
response = client.ask(request, request_dir, progress)    # blocking; ask_async() for asyncio
response.parsed        # an instance of `schema`, or None (see raw_text)
response.usage         # Usage(input / output / cache tokens)
response.record()      # what is stored in metadata files ("llm" block, round records)
```

The video describer (`describe`) and the chat agent (`chat.agent`) are
written against this interface only. Pictures are always
`ImagePart(core.images.fit_jpeg(data))`. A provider implements two methods:
`write_request()` writes the exact request in its wire format into
`request_dir` (`llm.request.*`, `llm.system.txt`) so it can be inspected and
replayed (also without sending it, e.g. `describe --dry-run`), and `_send()`
makes one attempt, raising `LLMThrottled` on a transient failure (rate /
usage limit, overload, 5xx, timeout) — `ask()` then waits
`LLM_RETRY_SLEEP_SEC` and retries, up to `LLM_MAX_RETRIES` times.

| `LLM_PROVIDER` | module | how it reaches the model |
|----------------|--------|--------------------------|
| `claude-cli` (default) | `ai/claude_cli` | `claude -p` with the system prompt only, no tools, from an empty folder; the answer schema as `--json-schema`; the user's Claude Code login |
| `claude-api` | `ai/claude_api` | the Anthropic Messages API (`pip install .[claude-api]`) with the same login's OAuth token |

Models: `DESCRIBER_MODEL` / `DESCRIBER_EFFORT` for clip descriptions,
`CHAT_AGENT_MODEL` / `CHAT_AGENT_EFFORT` for notifications and chat rounds,
`CHAT_AGENT_MEMORY_MODEL` / `CHAT_AGENT_MEMORY_EFFORT` for rounds that change
memory.

### Adding a provider (e.g. Gemini)

For the describer and the chat agent (both send text + still images):

1. `ai/gemini/client.py`: `GeminiClient(LLMClient)` with `name = "gemini"`;
   `write_request()` builds `contents` from the parts (`TextPart` -> text,
   `ImagePart` -> inline JPEG bytes) with `request.schema` as the response
   schema (JSON output) and writes it to `request_dir`; `_send()` sends it,
   maps the SDK's usage onto `Usage`, and raises `LLMThrottled` for 429 / 5xx
   / timeouts (`ai/transient.py` recognises the usual error texts).
2. Settings: add `"gemini"` to `LLMProvider` in `settings/ai.py` and a
   `GEMINI_API_KEY` (`# @secret` in `.env.example`); `make_llm()` returns the
   client for it. Models are chosen with the existing `DESCRIBER_MODEL` /
   `CHAT_AGENT_*` keys.
3. `pip install .[gemini]` (google-genai).

`ai/gemini/describer.py` is a different, reference use of Gemini: a model
that watches *video* can be given the trimmed clip itself instead of sampled
frames. It is not wired in; its docstring lists what wiring it would take
(a describer interface shared with `describe.VideoDescriber`, and a choice
between them in `pipeline.process`).

## Settings

`settings.Settings` is a pydantic-settings class composed of sections, each
next to the code that owns it: `storage` (STORAGE_ROOT, storage areas, time
zone), `cameras` (CAMERA_PROVIDER, CAMERAS, filters), `processing` (video
processor, YOLO), `ai` (LLM provider, describer model), `chat` (chat agent,
Telegram), `webui`, `stitch`, and each camera provider's own section
(`providers/blink/settings.py`). Values come from the environment, then
`.env` at the project root (`core.paths.ENV_FILE`), then the defaults.
Services call `load_settings()` once; library code that needs a value without
a `Settings` at hand uses `get_settings()` (cached).

`.env.example` documents every key — groups, help text, which services need a
restart, and tags for the configuration page (`settings/document.py`, see
[webui.md](webui.md#configuration-page)).

## Storage

Every file goes through a storage area (`core/storage.py`): `make_storage(settings,
AREA)` returns a `Storage` addressed by keys (forward-slash paths relative to
the area), so a clip store could move to S3 without changing its callers.
Areas: `RAW_VIDEOS`, `PROCESSED_VIDEOS`, `IGNORED_VIDEOS`, `TEMP_VIDEOS`,
`LIVE_SNAPSHOTS`, `CHAT`, each with `<AREA>_STORAGE_TYPE` (only `localdisk`
today) and `<AREA>_PREFIX` (its folder under STORAGE_ROOT). Next to them,
`core/runtime.py` keeps `state/`, `logs/` and `models/`. File naming and
formats: [data_formats.md](data_formats.md).

Clip identity: a clip is `<YYYY-MM-DD-HH-MM-SS>` (local recording time) under
`<date>/<Camera_Name>/` (`camera_folder()`: spaces and `/` become `_`); across
cameras it is `<stamp>-<CAMERA_NAME>` (`clip_name()`), which is also the event
id. The camera's real name always travels in the records (`"camera"`), never
reconstructed from a folder name.

## Logging

`core.logs.setup_logging(program, verbose, *loggers)`: one line per event,
`2026-10-18 09:12:03 INFO    [process] Person found`, to stdout and to
`STORAGE_ROOT/logs/<program>.log` (rotated at 10 MB). The named loggers log at
INFO, every other module only warnings unless `-v`. Also in `logs/`:
`chat_prompts.log` (every chat agent request in full), `run.log` (the
supervisor), `hw_trace.log` / `hw_monitor.log` (GPU diagnostics).
