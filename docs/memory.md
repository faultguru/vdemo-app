# Memory

vdemo has two memory stores. The split is intentional: durable identity
facts in one, everything else in per-day files that expire automatically.

← [Back to overview](README.md)

## Two stores at a glance

| | Persistent | Daily |
| --- | --- | --- |
| File | `state_dir/memory.md` | `state_dir/daily/YYYY-MM-DD.md` |
| Class | `Memory` (`blink_watcher/memory.py`) | `DailyContext` (`blink_watcher/daily.py`) |
| Scope | Survives forever until removed | Expires the moment we stop reading that day |
| Default for writes | RARE — reserved for identity-level facts | PREFERRED for almost everything |
| Read path | Injected into every system prompt as `## Current memory` | Active day injected as `## Day notes (<date>)`; other days via `get_day_notes(date)` tool |
| Write tools | `remember`, `forget` | `note_today`, `clear_today` |

The rule the prompts enforce: **prefer daily over persistent**.
Persistent is only for facts that would still be true in a month
("I am 45 years old", household roster, permanent notification
preferences). Anything tied to a specific video, day, or observation
goes to daily.

## Persistent memory — `memory.py`

Plain markdown file with a header and bulleted lines:

```
# Chat agent long-term memory

- (2024-06-10) household vehicle: blue sedan
- (2024-06-10) never alert about wind-blown foliage or shadows
- (2024-06-10) identify household members by stable physical traits, not daily clothing
```

`Memory` is a three-method class:

- `read()` — return the whole file.
- `read_body()` — return the body minus the header (what gets
  injected into the system prompt under `## Current memory`).
- `remember(note)` — lock, append a dated bullet, release. The
  date is the wall-clock day at time of save.
- `forget(substring)` — lock, drop any bullet whose text contains
  the substring (case-insensitive), write back, release.

Hand-editable — the file format is intentionally simple markdown
so you can open it in a text editor and fix things up directly.

## Daily memory — `daily.py`

One file per day: `state_dir/daily/YYYY-MM-DD.md`. Format is a
series of `[HH:MM] <text>` lines. Rollover is implicit — a new day
means a new file name, and the old file simply stops being read.
Old files are **not** deleted automatically so you can review
yesterday's scratch state.

```
[09:12] waiting for an afternoon delivery
[10:22] alerted: adult, grey jacket, dark jeans, carrying toolbox at side of house
[10:24] identified: adult, grey jacket, dark jeans, carrying toolbox = household member (household)
[14:08] package dropped on front porch — small cardboard box
```

`DailyContext` methods:

- `read_body(max_lines=30, date=None)` — read a day's file, return
  the most recent `max_lines`. Defaults to today's date. Used by
  `_system_with_memory` to fetch the active day's notes.
- `note(text, date=None, time=None)` — append. When `time` is
  omitted it defaults to wall clock; pass the event's real HH:MM
  to keep chronological order when processing backlog.
- `clear(substring, date=None)` — remove lines containing the
  substring. Deletes the file if it becomes empty.

Every method takes `date` so the same class serves "today's
scratchpad" and "backfill corrections into an older day's file".

## The active day

`ChatAgent._latest_event_date()` returns
`EventTools.list_event_dates()[-1]` — the newest `events.*.json`
file in the events directory. This is the single source of truth
for **where daily writes go**, regardless of what the LLM has been
looking at during a tool loop.

```
note_today(text="alert-on: afternoon delivery")
  → DailyContext.note(text, date=self._latest_event_date())
  → state_dir/daily/2024-06-15.md  (if 2024-06-15 is latest)
```

The same active day is what gets pre-fetched into the system prompt
under `## Day notes (<date>)`, so reads and writes are aligned. See
[prompt_construction.md](prompt_construction.md) for how the
injection works.

If the events directory is empty at call time, the write falls
back to the wall-clock day.

## Canonical line formats in daily files

The prompts establish three structured prefixes the LLM uses
consistently so other parts of the pipeline can match on them.

### `identified: <trait signature> = <identity>`
Written by the chat agent when the user corrects a mis-identification
(for example, correcting a mis-identified visitor). The trait
signature must be concrete (clothing, build, carried object,
activity). When the identity is a household member, the literal
suffix `(household)` is appended so the notifier can apply the
household-SKIP rule by plain text match.

Example:
```
identified: adult, grey jacket, dark jeans, carrying toolbox = household member (household)
```

### `alerted: <trait signature> <action> at <location>`
Written by the notifier *every time* it sends an alert. Records a
canonical signature of the subject so a recurrence of the same
event can SKIP as a duplicate. Uses the same trait format as
`identified:` so user corrections can match against it.

Example:
```
alerted: adult, grey jacket, dark jeans, carrying toolbox walking along side of house
```

### `alert-on: <subject / action / location>`
Written by the chat agent when the user asks for a one-day-only
opt-in ("let me know if you see them again"). Treated by the
notifier as a HARD OVERRIDE — if the current event matches an
`alert-on:` line, alert the user even if other rules would SKIP.
Expires with the day.

Other line shapes the prompts may emit (less structured, still
useful for correlation):

- `waiting for <delivery>` — user expectation.
- `package dropped on <location> — <description>` — tracked open
  situation, cleared by `clear_today` on pickup.
- `person inside: <description>` — tracked visitor.

## Decision tree the notifier follows

Step 0 of the notifier's compose pipeline (in `chat_notify.system.txt`)
is an identification lookup against `## Day notes`:

```
for line in day_notes:
    if line starts with "identified:" and traits match current event:
        if identity ends with "(household)":
            emit SKIP: identified as <identity>, routine household activity
        else:
            mention the identity by name, FRIENDLY tone, maybe SKIP
```

Only if no `identified:` line matches does the notifier fall
through to the `alerted:` duplicate check, then the routine
household-member check, then tone selection and alert generation.
`alert-on:` opt-ins pre-empt everything, even identification
matches.

## Hand-editing

Both files are plain text and safe to hand-edit. Useful moves:

- Open `state_dir/memory.md` and delete a stale fact, or rewrite
  one in place. No lock needed — the file is read on every
  `_system_with_memory` call, so next turn sees the change.
- Open today's `state_dir/daily/YYYY-MM-DD.md` to inspect the
  scratchpad the LLM is reasoning over. Delete noise lines to
  clean up the active state.
- Tail an older daily file to audit what the LLM saw and wrote on
  that day — none of it is purged automatically.

The only caveat: `DailyContext` caps `read_body` at 30 lines, so a
very chatty day will have its earliest entries fall off the active
view. Everything is still on disk — the cap only affects what gets
injected into the prompt.

## See also

- [chatbots.md](chatbots.md) — which write tools exist in the chat
  vs notify configurations.
- [prompt_construction.md](prompt_construction.md) — how the two
  stores get injected into the system prompt on every call.
- [examples.md](examples.md) — real-world examples of daily and
  persistent memory contents alongside the event JSON that
  produced them.
