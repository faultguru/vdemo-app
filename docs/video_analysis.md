# Video Analysis Pipeline

`blink_watcher.describe_video` turns a Blink surveillance clip into a structured
metadata JSON plus a thumbnail and a folder of important keyframes.

## What it does

```
.mp4 ──► OpenCV motion detect ──► YOLOv8 object detect ──► Gemini / Claude ──► .json
                                                                           ├─► <stem>.jpg   (thumbnail)
                                                                           └─► <stem>.keyframes/
```

Four sequential stages, each cached in the output JSON so a rerun only repeats
what it has to:

1. **OpenCV motion detection** (`detect_segments`)
   - MOG2 background subtraction, downscaled to `MOTION_MAX_SIDE=960` for speed
   - Groups consecutive moving frames into `MotionSegment`s (with gap-merging)
   - Filters noise: min contour area, min frame area, and a `MAX_FRAME_AREA_RATIO=0.15`
     cap to drop full-frame lighting/exposure flashes (Blink's auto-exposure kick)
   - Picks `KEYFRAMES_PER_SEGMENT=8` evenly-spaced moving frames per segment
   - Also computes centroid-tracked motion direction as an internal ground truth

2. **YOLOv8 object detection** (`detect_objects`)
   - Runs `yolov8n.pt` on each OpenCV-picked keyframe
   - Filters to a movable-subject class whitelist (`person`, `car`, `dog`, …)
     — static scenery like `tv`, `chair`, `potted plant` is dropped
   - Validates each detection actually *moves* across keyframes (≥3% of frame
     side) so parked cars inside the motion bbox don't become false positives
   - If a segment has zero YOLO detections, its centroid direction is cleared
     and the LLM call is skipped entirely for that clip

3. **LLM description** (`describers.py`)
   - Provider-abstracted: `VIDEO_DESCRIBER=claude` or `gemini`
   - Gets motion segments + YOLO ground-truth classes as input
   - Returns the same `VideoAnalysis` pydantic schema regardless of provider:
     `overall_summary`, per-event `object/details/description`,
     `important_frames_sec`, and `thumbnail_sec`
   - Short-circuited if no YOLO detections exist — emits `"No motion."`

4. **Thumbnail + keyframe extraction**
   - Uses the LLM-picked `important_frames_sec` to seek the raw video and
     dump JPEGs to `<stem>.keyframes/`
   - Saves the LLM-picked `thumbnail_sec` frame as `<stem>.jpg` next to the video
   - Falls back to a local heuristic (prefer person → biggest bbox → middle
     keyframe) only when the LLM didn't run (no-motion / motion-only clips)

## Providers

Both providers implement `describers.VideoDescriber` and emit the same schema.

### Gemini (default, recommended)
- Native video input via the Gemini Files API — no keyframe sampling
- Trims to `[first_motion_start - 0.5s, last_motion_end + 0.5s]` before upload
  to reduce billed token count
- Adaptive sampling: **5 FPS** when any segment has a YOLO `person` detection,
  **1 FPS** otherwise (behavioral actions like "dropping a package" need finer
  temporal resolution than vehicle pass-throughs)
- Auth: `GEMINI_API_KEY` from `.env`
- Model: `GEMINI_MODEL` (default `gemini-2.5-flash`, tested with `gemini-3-flash-preview`)

### Claude (alternate)
- Sends cropped keyframe JPEGs (cropped to the motion bbox + 15% padding)
- System prompt + full keyframe set use prompt caching (`cache_control: ephemeral`)
- Adaptive thinking is enabled automatically on `opus-4-6` / `sonnet-4-6`
- Auth: OAuth token from `~/.claude/.credentials.json` (Claude Code's credentials)
- Model: `CLAUDE_MODEL` (default `claude-haiku-4-5`)

## Prompt files

Human-editable, stored in `prompts/`. Lines starting with `#` are stripped at
load time and never sent to the model.

- `describe_video.system.txt` — shared system prompt. Contains the
  **destination rule** that forbids guessing where a carried object ends up
  unless the before/after evidence makes it literal (prevents "package was
  placed inside the container" hallucinations when the package actually went
  on the ground beside it).
- `describe_video.claude.user.txt` — Claude user-turn instructions. References
  keyframe labels (`@ t=X.XXs`) because Claude only sees image crops.
- `describe_video.gemini.user.txt` — Gemini user-turn instructions. References
  the full-video time range with per-segment motion spans because Gemini
  ingests the raw clip.

Key rules that both user prompts enforce:
- No compass/directional phrases ("left to right", "enters from the top")
- No stationary-object mentions in the summary ("the car remains parked")
- No generic vehicle types ("pickup", "SUV") unless the body style is
  unmistakable — fall back to "car" / "vehicle"
- When a `person` is detected, describe the concrete action (dropping a
  package, carrying a box, checking a phone) — not just "walks through"
- Populate `important_frames_sec` (2–6 distinct action moments) and pick one
  as `thumbnail_sec`

## Environment variables

From `.env`:

```
VIDEO_DESCRIBER=gemini            # claude | gemini
CLAUDE_MODEL=claude-haiku-4-5
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_API_KEY=...
```

`VIDEO_DIR=./videos` governs where clips live on disk (for the downloader —
not used by `describe_video.py` directly).

## How to run it

### Single clip
```bash
python -m blink_watcher.describe_video videos/tests/2026-04-14-16-16-59.mp4
```

Outputs (next to the `.mp4`):
- `2026-04-14-16-16-59.json`          — full metadata (see schema below)
- `2026-04-14-16-16-59.jpg`           — LLM-picked thumbnail
- `2026-04-14-16-16-59.keyframes/`    — LLM-picked important frames as JPEGs

### Flags
- `--output <path>` — write the JSON somewhere other than next to the video
- `--motion-only`   — run OpenCV only, skip YOLO + LLM (fast sanity check)

### Cached reruns
The pipeline reuses prior processor output from the output JSON so you don't
burn time/money redoing stages that haven't changed:

- `opencv` — reused whenever the JSON already exists. Delete the JSON to force
  a fresh motion pass.
- `yolo` — reused when present in the JSON's `processors` array.
- `llm` — always re-run (no caching) so prompt / model changes take effect.

To force a full refresh: `rm videos/tests/<clip>.json` and rerun.

## Test clips

`videos/tests/` has hand-picked clips for regression testing:

| Clip | What it tests |
| --- | --- |
| package-drop clip | Package drop-off — person bends next to an outdoor container and leaves a package on the ground beside it. Tests the destination rule (must say "on the ground next to the container", not "inside the container"), person-at-5-FPS sampling, and important-frame picking across multiple motion segments. |
| walk-across clip | Simple walk-across — single segment, person walks down the steps. Tests the basic path with one YOLO detection. |
| no-motion clip | No real motion (minor lighting change). Tests the short-circuit: OpenCV finds a segment, YOLO finds nothing, LLM is skipped, summary is `"No motion."`. |

### What a good output looks like (package-drop clip)

```json
{
  "overall_summary": "A delivery person wearing a dark uniform walks onto a patio carrying a package and a phone. They bend down to leave the package on the ground next to an outdoor container before checking their phone and walking away.",
  "thumbnail_sec": 24.5,
  "important_frames_sec": [15.8, 24.5, 27.8, 29.4],
  "events": [
    {"segment_index": 1, "object": "person", "description": "A person walks onto the patio carrying a package and a phone, then bends down to place the package on the ground next to an outdoor container."},
    {"segment_index": 2, "object": "person", "description": "A person stands up, checks their phone, and walks away."}
  ]
}
```

Checks to run when evaluating a new prompt / model:

- [ ] `overall_summary` does not mention any stationary object (parked cars,
      unchanged scenery)
- [ ] No compass directions ("left to right", "upward", etc.)
- [ ] No frame entry/exit language ("enters from the left", "off-screen")
- [ ] Person actions are concrete ("placing a package", "checking a phone"),
      not generic ("walks through")
- [ ] The destination rule holds: "on the ground" vs "inside the container"
      is handled correctly based on the before/after evidence
- [ ] Vehicles use generic nouns unless the body style is unmistakable
- [ ] `important_frames_sec` has 2–6 distinct moments; `thumbnail_sec` is one
      of them; both are within `[0, duration_sec]`
- [ ] No-motion clip emits `"No motion."` and skips the LLM call

## Output JSON schema (top-level)

```jsonc
{
  "video": "tests/2026-04-14-16-16-59.mp4",
  "info": { "fps": 30.005, "frame_count": 902, "width": 1920, "height": 1080, "duration_sec": 30.062 },
  "processors": [
    { "name": "opencv", "completed_at": "...", "duration_ms": 19340 },
    { "name": "yolo", "model": "yolov8n.pt", "duration_ms": 9885, "classes_detected": ["person"] },
    { "name": "llm", "provider": "gemini", "model": "gemini-3-flash-preview", "duration_ms": 31500, "usage": { "input_tokens": 11435, "output_tokens": 282, "total_tokens": 11717 } }
  ],
  "overall_summary": "...",
  "thumbnail_sec": 24.5,
  "important_frames_sec": [15.8, 24.5, 27.8, 29.4],
  "events": [
    { "segment_index": 1, "start_sec": 14.7, "end_sec": 27.8, "bbox": [170,0,1502,1080], "object": "person", "details": "...", "description": "..." }
  ],
  "motion_segments": [ /* raw OpenCV output: bbox, keyframe times, direction, YOLO detections */ ]
}
```

## Tuning knobs (in `describe_video.py`)

| Constant | Effect |
| --- | --- |
| `WARMUP_FRAMES=35` | Frames fed to MOG2 before output is trusted. Higher = more tolerant of Blink's exposure flash at clip start. |
| `MIN_CONTOUR_AREA_RATIO=0.0015` | Smallest contour that counts as motion (0.15% of frame). Lower picks up distant subjects, raises false positives from rain/leaves. |
| `MAX_FRAME_AREA_RATIO=0.15` | Cap that drops segments where motion covers >15% of the frame at peak — catches global lighting flashes. |
| `MERGE_GAP_SEC=1.0` | Stitch segments separated by gaps smaller than this. |
| `MAX_SEGMENTS_TO_DESCRIBE=12` | Cap LLM input size. |
| `KEYFRAMES_PER_SEGMENT=8` | Number of evenly-sampled moving frames per segment the Claude path sends. Gemini ignores this (uses native video). |
| `KEYFRAME_MAX_SIDE=1600` | Max JPEG dimension sent to Claude. |
| `YOLO_MOVING_CLASSES` | Whitelist of COCO classes allowed to be reported as moving subjects. |

## Known weak points

- **OpenCV seek by frame number can be off by a frame or two** on some
  containers (MPEG-4 with variable frame intervals). The thumbnail may land
  on a neighboring frame of the one Gemini actually intended.
- **Prompt caching for Claude doesn't help across different clips** — the
  image content is the cache breakpoint, so reruns on the *same* clip are
  cheap, but each new clip pays full price.
- **Gemini's `temperature=0` is still not deterministic** for ambiguous
  scenes (e.g. package-near-container). The destination rule in the system prompt
  is the primary defense — updating it there is the right lever when output
  drifts.
