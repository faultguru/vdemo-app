# Files on disk

Everything lives under `STORAGE_ROOT` (default `./data-storage`). The files
are plain JSON and Markdown, written atomically (a `.part` file renamed into
place), and are the interface between the services — so their formats are a
contract: fields are added, never renamed or removed. Times are local with
their UTC offset (`2026-10-17T11:22:04-07:00`) unless named `*_at` in a
processor record (UTC). Paths inside records are relative to STORAGE_ROOT and
start with the area's folder (`videos.processed/...`).

```
data-storage/
  videos.raw/<date>/<Camera_Name>/<stamp>.mp4             clip as downloaded            (fetch)
                                 /<stamp>.metadata.json   clip record                   (fetch)
  videos.processed/inbox/<stamp>-<CAMERA>.metadata.json   queue of the video processor  (fetch -> process)
  videos.processed/<date>/<Camera_Name>/<stamp>.mp4       trimmed clip                  (process)
                                       /<stamp>.jpg       thumbnail
                                       /<stamp>.metadata.json   clip record + analysis + "llm"
                                       /<stamp>.event.json      the event
  videos.ignored/<date>/<Camera_Name>/<stamp>.metadata.json     dropped: record + "ignored"
  videos.temp/<date>/<Camera_Name>/<stamp>/               work folder while processing
                                          /event/         notifier work folder (round-<n>/, result)
  videos.live/<date>/<Camera_Name>/<stamp>.jpg            live monitor snapshots        (webui)
  chat/events-inbox/<stamp>-<CAMERA>.event.json           queue of the notifier         (process -> notify)
  chat/<date>/events.json                                 the day's events + decisions  (notify, chat-bot)
  chat/<date>/memory.md                                   day notes                     (chat agent)
  chat/<date>/chat_history.<chat id>.json                 chat window                   (chat agent)
  chat/memory/memory.md                                   long-term memory              (chat agent)
  chat/outbox/<stamp>-<CAMERA>.message.json               messages to send              (notify -> chat-bot)
  chat/.notify_state.json                                 when media was last offered
  state/cameras.json                                      camera catalog
  state/blink/session.json                                Blink session (tokens: keep private)
  logs/<program>.log, logs/chat_prompts.log, logs/run.log
  models/yolov8s.pt (+ ONNX export for DirectML)
```

Names: `<stamp>` is the local recording time `YYYY-MM-DD-HH-MM-SS`;
`<Camera_Name>` is the camera name with spaces and `/` as `_`
(`core.storage.camera_folder`); `<stamp>-<CAMERA>` is the stamp plus the
upper-cased folder name (`clip_name`), unique across cameras and used as the
event id. Folder names are only for humans — every record carries the real
camera name in `"camera"`.

## Clip record — `<stamp>.metadata.json`

Written by fetch (`pipeline/clip_metadata.py`), provider-neutral:

```json
{
  "version": 1,
  "camera": "Front Door",
  "date": "2026-10-17",
  "recorded_at": "2026-10-17T11:22:04-07:00",
  "original": {
    "provider": "blink",
    "provider_clip_id": "17529094602",
    "duration_sec": 30.03,
    "trigger": "motion",
    "provider_detections": ["person"],
    "folder": "videos.raw/2026-10-17/Front_Door",
    "video_file": "2026-10-17-11-22-04.mp4"
  }
}
```

`trigger` is `motion` / `snapshot` / `live_view` or the provider's own value;
`provider_detections` are the provider's object labels (`[]` if none). A clip
skipped by its camera's filter (`CAMERA_FILTERS`) has no `.mp4` and gets
`"download": {"status": "filtered", "filter", "reason", "checked_at"}`.

The video processor copies it into the processed (or ignored) area and adds:

| key | |
|-----|-|
| `processed` | `{"folder", "video_file"}` of the trimmed clip (kept clips only) |
| `info` | the trimmed clip: `fps`, `frame_count`, `width`, `height`, `duration_sec` |
| `motion_segments` | `[{index, start_sec, end_sec, duration_sec, bbox, peak_area_ratio, keyframe_times_sec, direction, centroid_start, centroid_end, detections}]`, on the trimmed clip's clock |
| `processors` | one record per step in run order — `triage`, `opencv`, `yolo`, `trim`, `llm` — each `{sequence, name, completed_at, duration_ms, ..., passed}`; `reused: true` when taken over from an earlier run |
| `llm` | the describer's answer (`summary`, `events`, `important_frames_sec`, `thumbnail_sec`, see [video_analysis.md](video_analysis.md#3-describe)) and the call: `provider`, `model`, `completed_at`, `duration_ms`, `frames_sent`, `stop_reason`, `retries`, `usage` (`input_tokens`, `output_tokens`, `cache_write_tokens`, `cache_read_tokens`, `cost_usd`) |
| `ignored` | dropped clips: `{"at", "by": "filter" / "triage" / "opencv" / "yolo" / "llm", "reason"}` |

## Event — `<stamp>.event.json`, `chat/events-inbox/`, `chat/<date>/events.json`

The video processor writes the event next to the processed clip and queues a
copy in `chat/events-inbox/`; the notifier adds it to the day's list
`chat/<date>/events.json` (a JSON array, oldest recording first, one entry
per `event_id`) together with its decision:

```json
{
  "event_id": "2026-10-17-11-22-04-FRONT_DOOR",
  "camera": "Front Door",
  "recorded_at": "2026-10-17T11:22:04-07:00",
  "video": "videos.processed/2026-10-17/Front_Door/2026-10-17-11-22-04.mp4",
  "metadata": "videos.processed/2026-10-17/Front_Door/2026-10-17-11-22-04.metadata.json",
  "thumbnail": "videos.processed/2026-10-17/Front_Door/2026-10-17-11-22-04.jpg",
  "original": {"folder": "videos.raw/2026-10-17/Front_Door", "video_file": "2026-10-17-11-22-04.mp4"},
  "processed": {"folder": "videos.processed/2026-10-17/Front_Door", "video_file": "2026-10-17-11-22-04.mp4"},
  "summary": {"description": "...", "people_types": ["delivery_person"], "actions": ["package_dropoff"],
              "people": [{"appearance": "...", "type": "delivery_person"}], "threat": {"level": "none"}},
  "track": {"id": "2026-10-17-11-22-04-FRONT_DOOR", "subject": "adult, blue cap, navy jacket", "same_as": []},
  "notification": {"decision": "send", "tone": "friendly", "message": "A package was just left at the front door at 11:22am.",
                   "skip_reason": null, "model": "claude-sonnet-5-5", "composed_at": "...",
                   "message_file": "outbox/2026-10-17-11-22-04-FRONT_DOOR.message.json", "sent_at": "..."}
}
```

`summary` is the describer's summary, trimmed to what the chat agent needs
(`pipeline.events.event_entry`). `track` links events of the same subject
(`same_as`: earlier event ids; linked events share the `id`). `notification`
is `decision` `send` / `skip` with the message or the `skip_reason`;
`sent_at` is set by the chat bot once Telegram has it. `security_mode` is the
security mode the clip was processed in (also in its `.metadata.json`). A clip
saved in HOME on an indoor camera, without analysis, has
`"fast": {"reason", "at"}`, a fixed summary and a `skip` notification; the
video processor writes it straight into the day's list (no `.event.json`, no
events-inbox copy, the notifier never sees it). Its `.metadata.json` has no
`processors`, `motion_segments` or `llm`, and its processed video is the
original clip. The day's list is changed by several services (processor,
notifier, chat bot), always under `<date>/events.json.lock`.

## Outgoing message — `chat/outbox/<stamp>-<CAMERA>.message.json`

```json
{"event_id": "2026-10-17-11-22-04-FRONT_DOOR", "date": "2026-10-17",
 "text": "A package was just left at the front door at 11:22am.",
 "photo": "videos.processed/2026-10-17/Front_Door/2026-10-17-11-22-04.jpg"}
```

Removed by the chat bot once sent.

## Chat state

- `chat/<date>/chat_history.<chat id>.json` — `[{"role": "user" | "model", "text": ...}]`,
  the last 10 turns; sent notifications are added as model turns.
- `chat/memory/memory.md` — long-term memory:

  ```markdown
  # Chat agent long-term memory

  - (2026-10-12) The household's car is a dark gray hatchback.
  - (2026-10-14) Do not notify about the gardener on Tuesdays.
  ```

- `chat/<date>/memory.md` — day notes, one `[HH:MM] <text>` line each:

  ```
  [09:05] expecting a grocery delivery this afternoon - notify when it arrives
  [11:22] alerted: delivery driver, navy jacket, dropped a package at the front door
  ```

- `chat/.notify_state.json` — `{"last_offer_ts": <epoch>}`: notifications
  offer pictures / videos at most once an hour.

## Work folders — `videos.temp/<date>/<Camera_Name>/<stamp>/`

While a clip is processed: `<stamp>.check.json` (step 1), `<stamp>.mp4` and
`<stamp>.analysis.json` (trim), the sampled frames + `manifest.json`, the LLM
request (`llm.request.*`, `llm.system.txt` — replayable by hand) and
`describe.result.json` (the answer with the call's details). The notifier's
`event/` subfolder holds each round's request (`round-<n>[-memory]/`) and its
result. Removed when the clip / event is done (kept with `--preserve`).

## State

- `state/cameras.json` — the camera catalog:
  `{"updated_at": "...", "cameras": [{"name", "provider", "id", "model", "has_own_person_detection"}]}`.
- `state/blink/session.json` — the Blink login session (tokens), so a
  restart needs no new login or 2FA. Treat it like a password.
- `state/security_mode.json` — the security mode:
  `{"mode": "normal" | "home" | "away" | "vacation", "since", "until" (HOME only), "by": "console" | "telegram" | "auto"}`;
  `state/security_mode.history.jsonl` — every change, one per line.
