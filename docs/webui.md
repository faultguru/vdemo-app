# The web console

`python -m blink_watcher webui` (default http://127.0.0.1:9090, `WEBUI_HOST`
/ `WEBUI_PORT`) serves two single-page apps from `blink_watcher/webui/static/`
and the JSON API they use. There is no login: keep `WEBUI_HOST=127.0.0.1`
unless the LAN is trusted; `.env` can only be changed from the PC itself.

- `/` (`index.html`) — the console: a day's events (or today's, live), the
  activities they form, the clips still in the pipeline, the live monitor and
  the chat with the analyst.
- `/settings` (`settings.html`) — the configuration page.

## Code

| module | |
|--------|-|
| `webui/server.py` | `make_app(settings)` (app factory) and `main()` (the `webui` command) |
| `webui/context.py` | `WebContext`, shared by every route: settings, time zone, the storage areas media may be served from, the stitcher, the camera list |
| `webui/cameras.py` | the one camera list (below) |
| `webui/incoming.py` | a day's clips still in the pipeline, from the storage areas |
| `webui/routes/events.py` | config, cameras, days, events (+ activities), metadata, media |
| `webui/routes/incoming.py` | incoming clips |
| `webui/routes/settings.py` | the configuration page's API (`settings.document`) |
| `webui/routes/live.py` | the live monitor (`live.LiveMonitor`) |
| `webui/routes/chat.py` | the console chat (`chat.agent.ChatAgent`, console mode) |

## API

| route | |
|-------|-|
| `GET /api/config` | today (local date), time zone, poll interval, stitching on/off, the enabled cameras' names |
| `GET /api/cameras` | every known camera with its `enabled` flag and detection `filter` |
| `GET /api/days` | dates with events, newest first: events, notifications sent, threats |
| `GET /api/events/<date>` | the day's events (`chat/<date>/events.json`, oldest first) and, with `STITCH_ENABLED`, its activities (`groups`, newest first) |
| `GET /api/incoming/<date>` | the day's clips still in the pipeline |
| `GET /api/metadata?path=<rel>` | a `.metadata.json` (paths relative to STORAGE_ROOT, as in the events) |
| `GET /media/<rel>` | a video or picture, with Range requests (playback starts at once, seeking works) |
| `GET /api/live/cameras` | the enabled cameras on the provider account (connects to the provider) |
| `GET /api/live/status?cam=..` | heartbeat of the live monitor -> the cameras' latest snapshots |
| `GET /api/live/stream/<camera>` | the camera's live view as MJPEG |
| `GET /api/chat`, `POST /api/chat`, `POST /api/chat/clear` | the console chat (multipart: `text`, `date`, `live`, `files`) |
| `GET /api/settings`, `POST /api/settings` | the configuration page: options with their `.env` values; `{"changes": {KEY: value or null}}` |
| `GET /api/security-mode`, `POST /api/security-mode` | the security mode (header dropdown): `{"state": {mode, since, until, by}, "headline", "modes"}`; `{"mode": "normal" / "home" / "away" / "vacation"}` |

Only files inside the media areas (raw, processed, ignored, temp videos, live
snapshots) are served; anything else, or any path leaving STORAGE_ROOT, is a
404. The web console writes nothing but the live monitor's snapshots,
`.env` and the security mode (`state/security_mode.json`).

The header's **HOUSE** dropdown shows and sets the security mode (see
[chat.md](chat.md#security-mode)); it refreshes every 30 s, so HOME ending at
`HOME_MODE_UNTIL` or a change from Telegram shows up. Events processed in
AWAY / VACATION carry a `⌂ away` / `⌂ vacation` tag; an indoor clip saved in
HOME without analysis is a card tagged `⌂ residents home` with a HOME badge
(thumbnail: the clip's middle frame). Such clips are stitched by time alone,
with each other only (see `stitch/stitcher.py`).

## Cameras

Every camera picker in the console and on the configuration page uses one
list, `webui/cameras.py`: the camera catalog (`providers.catalog`,
`STORAGE_ROOT/state/cameras.json`, written whenever the fetch daemon or the
live monitor lists the account's cameras). Until a catalog exists, the names
in `CAMERAS` (unless it is `all`) and the cameras seen in the last 14 days'
events stand in. Each camera carries its flags from the current settings:
`enabled` (in `CAMERAS`, or `CAMERAS=all`) and `filter` (its `CAMERA_FILTERS`
entry: `always` / `person` / `person_or_vehicle`, default `person`).

## Events and activities

The day view lists the day's events with their descriptions, the
notification decision and the threat level; a click opens the clip with its
full analysis (each pipeline stage's result, motion segments, the frames the
LLM marked important). With `STITCH_ENABLED`, events are grouped into
**activities** (`stitch/stitcher.py`, no AI): two clips close in time are
linked when one rule holds —

| link | rule |
|------|------|
| overlap | another camera recording at the same moment (`STITCH_OVERLAP_SLACK_SEC`), similar people / labels / description (`STITCH_MIN_SCORE_OVERLAP`) |
| continuation | the same camera's next clip within `STITCH_CONTINUATION_GAP_SEC` (`STITCH_MIN_SCORE_CONTINUATION`) |
| transition | another camera starting within `STITCH_TRANSITION_GAP_SEC` after (`STITCH_MIN_SCORE_TRANSITION`; easier between `STITCH_CAMERA_NEIGHBORS`, harder between others) |
| track | the notifier said it is the same subject (`track` in the event), within `STITCH_TRACK_GAP_SEC` |

The similarity compares people broken into attributes (colour + garment,
hair, type, carried objects — word lists in `stitch/vocab.json`), labels /
actions / vehicles, and description words. An activity longer than
`STITCH_MAX_SPAN_SEC` is cut at its largest gap. Nothing is stored: activities
are recomputed on every request, so changing a setting regroups every day.
`python -m blink_watcher stitch --date <day>` prints a day's activities with
each link's rule and score, for tuning. An activity plays as one sequence,
optionally picture-in-picture with the other cameras recording at the same
moment.

## Incoming clips

Today's view also shows the clips still on their way (`webui/incoming.py`,
read from the storage areas, nothing written), each with the step it is in:

| stage | seen as |
|-------|---------|
| queued | `videos.processed/inbox/<stamp>-<CAMERA>.metadata.json`, no work folder yet |
| analyze | the work folder `videos.temp/<date>/<Camera>/<stamp>/` exists (triage, OpenCV, YOLO) |
| trim | `<stamp>.check.json` written |
| frames | `<stamp>.analysis.json` written: sampling key frames |
| describe | the describer's request files written (`llm.request*`): the LLM is analyzing the frames |
| save | `describe.result.json` written: moving the results out |
| notify | `chat/events-inbox/<stamp>-<CAMERA>.event.json`: awaiting the notification decision |
| ignored | dropped in the last 3 minutes (`videos.ignored/...`), with who dropped it and why |

A work folder unchanged for 3 minutes is shown as stalled (failed, retried
later). A clip leaves the strip when its event appears in the day's list.

## Live monitor

`live/` on the camera-provider API. While the page shows the live monitor it
calls `/api/live/status` every few seconds (the heartbeat); as long as
heartbeats come, a loop asks the provider for a new picture of each camera on
screen, as often as the provider allows (Blink: one camera command every
~2 s). Snapshots are saved in the `LIVE_SNAPSHOTS` area (`videos.live/`) and
kept `WEBUI_LIVE_KEEP_HOURS`. A tile's live view (`/api/live/stream/<camera>`)
runs the provider's live stream through ffmpeg into JPEG frames shared by
every viewer; it stops shortly after its last viewer leaves. The provider
session is opened on first use only, and refreshes the camera catalog.

## Chat

The console's chat panel talks to a console-mode chat agent
([chat.md](chat.md#the-web-consoles-chat)): with a day selected the questions
are about that day; attached pictures (and a few frames of an attached video)
are shown to it but never stored; with the live monitor open the message
names the cameras on screen and the agent also gets each camera's latest
snapshots (`WEBUI_LIVE_AI_FRAMES` from the last `WEBUI_LIVE_AI_WINDOW_SEC`).
The conversation lives in the server's memory only.

## Configuration page

`settings/document.py` builds the page from `.env.example`: every documented
option with its help text, its type and default (from `Settings`), its
current `.env` value (secrets only as set / not set), grouped as in the file.
Saving validates the whole new `.env` against `Settings` first, then rewrites
only the changed lines (comments and order kept; the previous file goes to
`.env.bak`), and names the services to restart.

`.env.example` conventions (see its header):

| line | meaning |
|------|---------|
| `# --- Name ---` | starts a group; comment lines right under it describe the group |
| `# @programs a, b` | under a group header: the services to restart after a change |
| comment lines above an option | its help text |
| `KEY=value` | an option and its example value; `# KEY=value`: off by default |
| `# @advanced` | shown only in advanced mode |
| `# @secret` | never shown, only replaced |
| `# @choices a\|b` | the allowed values, when the code does not fix them |
| `# @default x` | the default, when it is not in `Settings` |
| `# @type cameras` | a list of camera names, picked from the camera list; `# @type cameras all`: may also be `all` |
| `# @type camera-map` | `{"Camera A": ["Camera B"]}` pairs of cameras, either direction (neighbours) |
| `# @type camera-filter` | `{"Camera A": "always"}` a detection filter per camera (`core.filters`); cameras not listed use the default (`person`) |
| `# @label Text` | a short name for the option (column header in the camera table) |

The camera options of a group are shown together as one table — a row per
camera, a column per option: a switch for a `@type cameras` option (Enabled),
a dropdown for a `@type camera-filter` option (Download: Always / Person /
Person or vehicle) — and an "all cameras" switch for the option tagged `all`. A
`@type camera-map` option is a picker: per camera, the other cameras as
toggles; a pair is shown on both cameras' rows.
