# Examples

Concrete representative contents of every file the pipeline produces and
every memory store the chatbots read and write. All names, dates,
clothing details, and personas below are **fabricated placeholders** —
the shapes and field names match real files but the content does not.
Use it as a reference for "what should this file look like?".

← [Back to overview](README.md)

## 1. Describer output — `events/<date>/<event_id>.json`

Produced by `describe_video.py` for each `.mp4` and written next to the
clip. `event_log.process_video` copies the `.mp4`, the describer JSON,
the thumbnail, and the keyframes into the event directory.

```json
{
  "video": "2024-06-15-10-22-18-00.mp4",
  "info": {
    "fps": 30.0,
    "frame_count": 900,
    "width": 1920,
    "height": 1080,
    "duration_sec": 30.0
  },
  "summary": {
    "description": "A person walks across the patio carrying a toolbox toward the side of the house.",
    "actions": ["person_walking"],
    "people": [
      {
        "appearance": "adult in a grey jacket",
        "appearance_details": "grey jacket, dark jeans, brown boots, baseball cap",
        "type": "adult",
        "estimated_age": 40,
        "actions": ["person_walking", "person_standing"],
        "objects": ["toolbox"]
      }
    ],
    "vehicles": [],
    "ignore": false
  },
  "processors": [
    {"name": "triage", "duration_ms": 2100, "changed_ratio": 0.14, "threshold": 0.005, "passed": true},
    {"name": "opencv", "duration_ms": 20000},
    {"name": "yolo", "model": "yolov8s.pt", "duration_ms": 17500, "classes_detected": ["person"]},
    {
      "name": "llm",
      "provider": "gemini",
      "model": "gemini-3-flash-preview",
      "duration_ms": 38000,
      "usage": {"input_tokens": 13000, "output_tokens": 180, "total_tokens": 13180}
    }
  ],
  "thumbnail_sec": 11.0,
  "important_frames_sec": [2.0, 5.0, 11.0, 13.0],
  "events": [
    {
      "segment_index": 0,
      "start_sec": 1.1,
      "end_sec": 29.9,
      "bbox": [60, 0, 1830, 1080],
      "object": "person",
      "details": "grey jacket, dark jeans",
      "description": "A person walks across the patio carrying a toolbox."
    }
  ],
  "motion_segments": [
    {
      "index": 0,
      "start_sec": 1.1,
      "end_sec": 29.9,
      "duration_sec": 28.8,
      "bbox": [60, 0, 1830, 1080],
      "peak_area_ratio": 0.29,
      "keyframe_times_sec": [1.1, 4.9, 8.8, 12.6, 16.5, 20.3, 25.0, 29.9],
      "direction": "downward",
      "centroid_start": [684, 508],
      "centroid_end": [636, 771],
      "detections": [
        {
          "class": "person",
          "confidence": 0.92,
          "bbox": [849, 0, 1033, 347],
          "present_in_keyframes": [0, 1, 2, 3, 4, 6, 7]
        }
      ]
    }
  ]
}
```

Key points:

- `summary` is the LLM-authored block that the event index copies into
  its compact form. The `ignore` flag lets the describer hint that a
  clip is uninteresting (e.g. triage-failed) so downstream suppression
  is easy.
- `processors` is the cache manifest — stages `triage`, `opencv`, and
  `yolo` are reused on rerun; `llm` is always re-run.
- `important_frames_sec` drives which JPEGs get extracted into
  `<event_id>.keyframes/`.
- `thumbnail_sec` is the single frame saved as `<event_id>.jpg`.

See [video_analysis.md](video_analysis.md) for the full schema and
tuning knobs.

## 2. Event index entry — `events/events.YYYY-MM-DD.json`

Written by `event_log.process_video` as a compact list. This file is
the read path for the chat agent's `list_events` tool.

```json
[
  {
    "event_id": "2024-06-15-10-22-18-00",
    "recorded_at": "2024-06-15T10:22:18-07:00",
    "video": "events/2024-06-15/2024-06-15-10-22-18-00.mp4",
    "metadata": "events/2024-06-15/2024-06-15-10-22-18-00.json",
    "thumbnail": "events/2024-06-15/2024-06-15-10-22-18-00.jpg",
    "summary": {
      "description": "A person walks across the patio carrying a toolbox toward the side of the house.",
      "actions": ["person_walking"],
      "people": [
        {
          "appearance": "adult in a grey jacket",
          "appearance_details": "grey jacket, dark jeans, brown boots, baseball cap",
          "type": "adult",
          "estimated_age": 40,
          "actions": ["person_walking", "person_standing"],
          "objects": ["toolbox"]
        }
      ]
    }
  },
  {
    "event_id": "2024-06-15-14-08-02-00",
    "recorded_at": "2024-06-15T14:08:02-07:00",
    "video": "events/2024-06-15/2024-06-15-14-08-02-00.mp4",
    "metadata": "events/2024-06-15/2024-06-15-14-08-02-00.json",
    "thumbnail": "events/2024-06-15/2024-06-15-14-08-02-00.jpg",
    "summary": {
      "description": "A delivery person leaves a small cardboard box on the front porch and walks back to a white van.",
      "actions": ["package_dropoff"],
      "people": [
        {
          "appearance": "delivery person",
          "appearance_details": "navy uniform shirt, dark shorts, running shoes",
          "type": "delivery_person",
          "estimated_age": 30,
          "actions": ["package_dropoff", "person_walking"],
          "objects": ["package"]
        }
      ],
      "vehicles": [
        {"appearance": "white van", "appearance_details": "", "actions": []}
      ]
    }
  }
]
```

Notes on shape:

- `event_id` format is `YYYY-MM-DD-HH-MM-SS-NN` where `NN` is a
  two-digit collision counter for events that land in the same second
  (`00` for the first event of each second).
- Path fields are relative to the repo root (they include the leading
  `events/`) and resolved — with sandbox checks — by
  `EventTools.resolve_under_events`.
- Summary people entries are heterogeneous: some use `age_group`, some
  use `type`, and optional fields like `estimated_age` may be present.
  The chat agent and notifier read whichever is available.
- There is no `camera` field — all clips come from the configured Blink
  camera set and the describer doesn't label them per-camera in the
  index.

## 3. Outbox handoff — `events/outbox/<event_id>.json`

One-shot handoff from the video pipeline to the notifier. Drained by
`chat_bot._process_outbox_file`.

```json
{
  "event_id": "2024-06-15-10-22-18-00",
  "prompt": "offer_media: true\nevent_id: 2024-06-15-10-22-18-00\nrecorded_at: 2024-06-15T10:22:18-07:00\npeople_types: [adult]\nactions: [person_walking]\ndescription: A person walks across the patio carrying a toolbox toward the side of the house.\ndetails: grey jacket, dark jeans, brown boots, baseball cap"
}
```

The `prompt` field is fed verbatim as the user-turn input to
`ChatAgent.compose_notification`. The `offer_media: true|false` header
tells the notifier whether it may invite the user to ask for a picture
(the chat agent suppresses this after recent media offers to avoid
spamming).

## 4. Persistent memory — `data/memory.md`

Identity-level facts only. Hand-editable.

```markdown
# Chat agent long-term memory

- (2024-06-10) Household: two adults and one school-age child.
- (2024-06-10) Household vehicle: blue sedan.
- (2024-06-10) Identify household members by stable physical traits (height, build, hair color) rather than daily clothing.
```

These facts are injected into every system prompt under
`## Current memory`. Note the convention of keying identity on **stable
physical traits** (height, hair color, build) rather than daily
clothing — a rule written into persistent memory precisely so the
model doesn't bind identity to something as ephemeral as a specific
jacket.

What does **not** go here:

- "That was me in the 10:22 clip" — goes to daily as an `identified:` line.
- "Let me know if you see that person again" — daily, one-day opt-in.
- "The package on the porch is the one I ordered" — daily, tracked situation.
- "I'm waiting for a delivery this afternoon" — daily, user expectation.

If a fact could plausibly be wrong next week, it belongs in daily.

## 5. Daily context — `data/daily/YYYY-MM-DD.md`

One file per day. Rollover is implicit (new date → new filename).
Old files are kept on disk for audit but only the active day (the
latest `events.*.json` on disk) is injected into the system prompt.

```
[10:22] identified: adult, grey jacket, dark jeans, brown boots, baseball cap, carrying toolbox = known visitor (contractor)
[10:45] alerted: known visitor (contractor), grey jacket, dark jeans, carrying toolbox at side of house
[--:--] ignore all other events for the rest of the day
[--:--] alert-on: afternoon delivery
[14:08] alerted: delivery person, navy uniform, dark shorts, dropping small cardboard box at front porch
[14:08] package dropped on front porch — small cardboard box
```

Things worth noticing in this sample:

- **`[--:--]` timestamps** appear on lines that weren't tied to a
  specific event — typically user instructions delivered via chat
  ("ignore other events today", "tell me when the delivery arrives").
  The `DailyContext.note` signature uses this placeholder when no
  event time is supplied and the target date isn't today.
- **`alert-on: afternoon delivery`** is a one-day HARD OVERRIDE. It
  fires at 14:08 when the delivery event matches — the notifier
  produces an alert even though a separate line says to ignore
  everything else today.
- **The 10:22 `identified:` line came from a user correction** ("that
  person is the contractor I hired"). The 10:45 `alerted:` line was
  then written by the notifier when a return trip fired anyway — the
  identification didn't SKIP because the contractor isn't household.
- **The `ignore all other events` line** is loose prose, not one of
  the structured prefixes. The notifier treats it as general SKIP
  discretion, separate from the `alert-on:` HARD OVERRIDE rule.

How each line got written:

| Line | Written by | Why |
| --- | --- | --- |
| `identified: ... = known visitor (contractor)` | chat agent via `note_today` | User replied to a 10:22 alert identifying the visitor. |
| `alerted: known visitor (contractor) ...` | notifier via `note_today` | Recorded automatically on the return-trip alert minutes later. |
| `ignore all other events for the rest of the day` | chat agent via `note_today` | User said "don't tell me about anything else today". |
| `alert-on: afternoon delivery` | chat agent via `note_today` | User added a single-day opt-in for the delivery they were expecting. |
| `alerted: delivery person ... cardboard box` | notifier via `note_today` | Written after the 14:08 delivery alert was delivered. |
| `package dropped on front porch — small cardboard box` | notifier via `note_today` | Tracked open situation from the same event. |

The canonical prefixes the notifier's decision tree keys off:

- **`identified: <traits> = <identity>[ (household)]`** — resolves a
  trait signature to a known person. `(household)` suffix triggers
  the notifier's household-SKIP rule. Step 0 of the compose pipeline.
- **`alerted: <traits> <action> at <location>`** — recurrence
  signature, written after every delivered alert, checked for
  duplicate suppression on the next similar event.
- **`alert-on: <subject>`** — one-day opt-in. HARD OVERRIDE — beats
  every SKIP rule for the rest of the day.

See [memory.md](memory.md) for the full story on these formats.

## 6. Sample notifier compose — input → output

Input (prompt passed to `compose_notification`, same shape as the
outbox file's `prompt` field), modelled on the 14:08 delivery event:

```
offer_media: true
event_id: 2024-06-15-14-08-02-00
recorded_at: 2024-06-15T14:08:02-07:00
people_types: [delivery_person]
actions: [package_dropoff]
description: A delivery person leaves a small cardboard box on the front porch and walks back to a white van.
details: navy uniform shirt, dark shorts, running shoes
```

System prompt includes `## Day notes (2024-06-15)` pre-loaded with the
daily-context sample above, including the earlier
`alert-on: afternoon delivery` line and the `ignore all other events`
line.

Notifier's decision trace:

1. **Step 0 — identification lookup.** Compare traits against
   `identified:` lines. No match (the only identification entry is
   for the contractor with a toolbox — very different traits). Fall
   through.
2. **HARD OVERRIDE (today only) check.** Scan for `alert-on:` lines.
   Match on `alert-on: afternoon delivery` vs the delivery
   description → forced alert, FRIENDLY tone. This beats the general
   `ignore all other events for the rest of the day` instruction.
3. Produce alert text referencing the earlier opt-in.
4. Call `note_today` to record the recurrence signature and the
   dropped-package open situation.

Final output:

```
<friendly> The delivery you were expecting was just dropped off on the front porch at 2:08pm.
```

Plus two side effects: an `alerted:` line capturing the delivery
person's trait signature, and a `package dropped on front porch —
small cardboard box` tracked-situation line.

Contrast case — if the `alert-on:` line didn't exist, Step 2 would
match the general SKIP instruction for today and the notifier would
emit:

```
SKIP: user asked to ignore all other events today
```

And `chat_bot._process_outbox_file` would log the reason and delete
the outbox file with no Telegram send.

## 7. Chat history — `data/chat_history/<chat_id>.json`

Sliding window of the last `HISTORY_WINDOW=10` user/model turns,
written by `ChatAgent._save_history`. Tool-call round-trips within a
single turn are NOT persisted — only the final user-visible text.

```json
[
  {
    "role": "user",
    "text": "don't bother me with anything else today, but let me know when my delivery shows up this afternoon"
  },
  {
    "role": "model",
    "text": "OK. I'll hold other alerts for the rest of the day and ping you as soon as the delivery arrives."
  },
  {
    "role": "model",
    "text": "[auto-notification event_id=2024-06-15-14-08-02-00] The delivery you were expecting was just dropped off on the front porch at 2:08pm."
  },
  {
    "role": "user",
    "text": "what happened this morning?"
  },
  {
    "role": "model",
    "text": "This morning (June 15):\n- 10:22 AM: A visitor in a grey jacket arrived carrying a toolbox and walked to the side of the house.\n- 10:45 AM: The same visitor was seen again on their way out."
  }
]
```

Note three things:

- **Auto-notifications are appended** with an
  `[auto-notification event_id=...]` prefix via
  `ChatAgent.append_notification`, so the next user message sees them
  as prior model turns and the chat agent knows "the notifier already
  told the user about this." Multiple consecutive `model` turns are
  normal when several alerts fire between user messages.
- **Relative-date questions** like "this morning" get resolved
  against the `## Current date` block injected into the system prompt,
  and the chat agent then calls `list_events` / `get_event_details`
  to produce the summary. None of that tool traffic is persisted —
  only the final natural-language reply.
- **After a successful `send_thumbnail` / `send_video`** the chat
  agent replies with the literal sentinel `SILENT`. That sentinel is
  persisted into history as-is, and `chat_bot.py` suppresses the
  redundant Telegram send.

## See also

- [video_analysis.md](video_analysis.md) — how the describer JSON in
  section 1 is produced.
- [event_loading.md](event_loading.md) — how sections 1, 2, and 3
  relate in the pipeline.
- [chatbots.md](chatbots.md) — how sections 3, 6, and 7 are wired
  together at runtime.
- [memory.md](memory.md) — deeper coverage of sections 4 and 5.
- [prompt_construction.md](prompt_construction.md) — how the active
  day in section 5 is chosen and injected into the system prompt.
