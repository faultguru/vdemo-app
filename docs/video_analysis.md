# Video analysis

The video processor (`python -m blink_watcher process`, `pipeline/process/`)
turns a downloaded clip into either an event or an "ignored" record. Most
camera clips show nothing worth a message (wind, light changes, a car passing
on the street), so the pipeline runs the cheapest checks first and only sends
the clips that survive them to the LLM.

```
 inbox/<stamp>-<CAMERA>.metadata.json
   |
   |  the camera's filter (CAMERA_FILTERS: always | person | person_or_vehicle) on the provider's detections
   |  security mode HOME + an INDOOR_CAMERAS camera: saved as it is (original video, thumbnail) and added to the day's events index; no event file, no notifier
   v
 1 check     triage -----> OpenCV motion ----> YOLO: a required class moving?      (vision.check)
   |           "nothing changes"  "no motion"     "no person moving"   -> ignored
   v
 2 trim      YOLO person scan over the clip, cut to while people move           (vision.tracks, vision.focus)
   v
 3 describe  sample key frames, crop to the activity, ask the LLM -> VideoAnalysis  (vision.sampling, describe)
   |           summary.ignore, no threat, no person (no vehicle on person_or_vehicle cameras) -> ignored
   v
 4 save      videos.processed/<date>/<Camera>/: <stamp>.mp4, .jpg, .metadata.json, .event.json
             chat/events-inbox/<stamp>-<CAMERA>.event.json
 5 clean up  inbox file and work folder removed
```

Everything a clip produces is first written to its work folder in the temp
area, `videos.temp/<date>/<Camera>/<stamp>/`, and moved out in the save step,
so a crash never leaves half-written results in the processed area; a failed
clip keeps its inbox file and is retried. The web console's "incoming" strip
shows a clip's progress from the files in that folder
([webui.md](webui.md#incoming-clips)).

## 1. Check (no AI)

`vision/check.py` runs three steps, each stopping the clip when it says no.
Each leaves a processor record in the metadata (`{"name", "completed_at",
"duration_ms", ..., "passed"}`).

- **triage** (`vision/triage.py`) — 6 frames spread over the clip, decoded at
  160 px, compared pairwise. If no pair differs in more than 0.5 % of its
  pixels, nothing happens in the clip. Costs well under a second.
- **opencv** (`vision/motion.py`) — MOG2 background subtraction on every 3rd
  frame at 960 px, after a warm-up that absorbs the camera's exposure
  adjustment; contours below 0.15 % of the frame (rain, leaves) and exposure
  flashes are ignored. Moving frames are grouped into motion segments (gaps
  under 1 s merged, blips under 0.4 s dropped), each with a bounding box, a
  direction and 8 keyframes.
- **yolo** (`vision/yolo.py`) — YOLOv8s on the segments' keyframes. A
  detection counts only if it lies in the segment's motion box and its center
  moves across the keyframes, so a parked car or a person-shaped decoration
  does not pass. The step passes when a `YOLO_REQUIRED_CLASSES` class
  (default `person`; `[]` = any moving class) moves. Weights are downloaded
  to `STORAGE_ROOT/models/` on first use; `YOLO_DEVICE=directml` runs an ONNX
  export on a DirectX 12 GPU (`hwtrace` writes crash-safe trace lines around
  that GPU work).

## 2. Trim

`vision/tracks.py` scans the whole clip with YOLO at 5 fps and links the boxes
into tracks (greedy IoU); `vision/focus.py` keeps the span while a person
track moves on screen — falling back to the motion segments' span — plus 1 s
on each side, and `vision/ffmpeg.py` cuts it (x264, audio kept). The trim
record (`removed_head_sec`, `offset_sec`, `duration_sec`, `focus`) maps the
trimmed clip's timeline back onto the original. The step 1 results are
shifted onto the trimmed clip's clock.

## 3. Describe

**Frame sampling** (`vision/sampling.py`, also the `frames` command): the LLM
gets still frames, not video, so they are chosen here, on the trimmed clip:

- a baseline of 1 frame per second across the activity span;
- bursts at 4 fps around the moments that matter for "what happened": a
  tracked person's box suddenly getting squatter (bending down — dropping off
  or picking up something), and each track's first and last sighting
  (arriving with / leaving with);
- at most 48 frames, each cropped to the activity region (all tracks' boxes,
  padded so the scene stays recognisable) when that saves at least 10 % of
  the frame, at most 960 px on the long side.

The frames and `manifest.json` (times, crop, trim, the tracks) go to the work
folder.

**The describer** (`describe/`): `describe.request` builds the request — the
system prompt `prompts/describe_video.system.txt`, then the user turn
`prompts/describe_video.user.txt` filled with the clip's facts (camera, time,
length), the security mode (`prompts/security_mode.txt`: rule 19 uses it for the
threat level and `ignore`, never for the description), how the frames were sampled (`describe_keyframes.input_mode.txt`,
`describe_keyframes.crop.txt`) and the motion segments with their YOLO
detections, followed by every frame as a timestamped image. `VideoDescriber`
sends it to the configured LLM provider (`LLM_PROVIDER`, `DESCRIBER_MODEL`,
`DESCRIBER_EFFORT`) and parses the structured answer as `VideoAnalysis`
(`describe/schema.py`):

- `summary`: a one-paragraph `description`; `people` (appearance,
  appearance details, type (kid, adult, delivery person, mail carrier, visitor, ...),
  estimated age, actions, objects carried, labels), `vehicles`, clip-level `actions` and
  `labels`; a `threat` assessment (`none` / `suspicious` / `threat`, signals,
  evidence); `ignore` when nothing in the clip matters.
- `events`: one entry per moving object per motion segment.
- `important_frames_sec` and `thumbnail_sec`, on the trimmed clip's timeline.

The enums in the schema (person types, actions, objects, threat levels) are a
contract with the notification and chat prompts; free-form `labels` carry
everything else. The request files (`llm.request.*`, `llm.system.txt`) and
the answer with its usage and timing (`describe.result.json`) stay in the work
folder. `python -m blink_watcher frames <clip.mp4>` writes a frames folder
(no LLM) and `python -m blink_watcher describe <frames folder>` describes it,
e.g. with another `--model` or `--provider`, or `--dry-run` to only write
the request.

A clip that was processed before reuses the earlier run's trim and
triage / OpenCV / YOLO results, so only the LLM runs again; `--redo` reruns
everything, `--preserve` keeps the inbox file and work folder (testing).

## 4. Save

A kept clip ends up in `videos.processed/<date>/<Camera>/`:

| file | contents |
|------|----------|
| `<stamp>.mp4` | the trimmed clip |
| `<stamp>.jpg` | thumbnail: the frame at the LLM's `thumbnail_sec` |
| `<stamp>.metadata.json` | the clip record from fetch + `processed`, `info`, `motion_segments`, `processors` (in run order), `llm` (the answer, model, timing, tokens, cost) |
| `<stamp>.event.json` | the event (`pipeline.events.event_entry`), also queued as `chat/events-inbox/<stamp>-<CAMERA>.event.json` for the notifier |

An ignored clip — filtered, nothing changes, no motion, no required object
moving, or `summary.ignore` from the LLM without a threat and without a person
in its answer (or a vehicle, on a `person_or_vehicle` camera; prompt rule 20:
look-alikes such as decorations are not people) — leaves only its
`.metadata.json`, with every processing detail and an `"ignored": {"at",
"by", "reason"}` record, in `videos.ignored/<date>/<Camera>/`.

Formats: [data_formats.md](data_formats.md).

## Tuning

- Settings: `CAMERA_FILTERS` (per camera; `always` for cameras without their
  own person detection), `PROCESS_LIMIT`, `YOLO_REQUIRED_CLASSES`,
  `YOLO_DEVICE`, `DESCRIBER_MODEL`, `DESCRIBER_EFFORT`, `LLM_RETRY_SLEEP_SEC`,
  `LLM_MAX_RETRIES`.
- Thresholds are settings too, in the "Video analysis" group of `.env.example`
  (advanced on the configuration page; `settings/vision.py`): the quick check
  `TRIAGE_*`, motion `MOTION_*`, YOLO `YOLO_CONF_THRESHOLD` /
  `YOLO_MOTION_OVERLAP` / `YOLO_MIN_SHIFT_RATIO`, tracking `TRACK_*`, frames
  for the AI `SAMPLE_*` and `BEND_*`, trimming `TRIM_*`, thumbnails
  `THUMBNAIL_*`. Each `vision/` module reads them into its constants at the top
  (e.g. `motion.MIN_SEGMENT_SEC` = `MOTION_MIN_SEGMENT_SEC`), each with a
  comment on what it guards against; a change applies when process restarts.
- `python -m blink_watcher frames <clip.mp4> -o <folder>` shows which frames a
  clip would send (and the trim) without calling the LLM.
