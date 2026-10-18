"""The video processor: runs the clips queued in the processed videos inbox
through the analysis pipeline, including the LLM description.

Reads <PROCESSED_VIDEOS>/inbox/<stamp>-<CAMERA>.metadata.json (queued by the
fetcher), oldest recording first, and handles up to PROCESS_LIMIT of them
(0 = all). Clips whose provider detections don't match their camera's filter
(CAMERA_FILTERS: always | person | person_or_vehicle) are skipped. For each
matching clip (in the home security mode, an indoor camera's clip is saved as
it is, with no step run: see ClipProcessor._run_fast), cheapest steps
first — everything a clip produces (cut video, metadata, frames, LLM request
and result) is written to its work folder in the temp area (TEMP_VIDEOS_*;
<YYYY-MM-DD>/<Camera_Name>/<stamp>/) and moved out in step 4:

  1. check    — vision.check on the original clip: triage (do sampled frames
                change?) -> OpenCV motion segments -> YOLO on the segments'
                keyframes (a moving YOLO_REQUIRED_CLASSES object, e.g. a
                person?). The first failure ignores the clip: no person scan,
                no trim, no LLM — most clips end here. -> <stamp>.check.json
  2. trim     — YOLO person scan + cut the quiet head / tail (vision.focus:
                while a person moves on screen, else the motion span; audio
                kept, original untouched). Step 1's results are shifted onto
                the trimmed clip's clock. -> <stamp>.mp4, <stamp>.analysis.json
  3. describe — the describer (blink_watcher.describe) samples frames from
                the trimmed clip, reusing step 2's scan, and sends them to the
                LLM (LLM_PROVIDER, DESCRIBER_MODEL). -> frames, manifest.json,
                llm.request.*, describe.result.json
  4. save     — cut video + metadata moved from the work folder to the
                processed area, same folder structure as the raw area
                (<YYYY-MM-DD>/<Camera_Name>/):
                  <stamp>.mp4            trimmed clip
                  <stamp>.jpg            thumbnail (frame at the LLM's thumbnail_sec)
                  <stamp>.metadata.json  the clip's metadata as queued (with
                      "original"), "processed": {"folder", "video_file"}, the
                      analysis (info and motion_segments on the trimmed clip's
                      clock, processors in run order with "sequence" 1..N) and
                      "llm": the LLM's answer (summary, events, thumbnail_sec,
                      important_frames_sec) with model, timing, tokens and cost.
                  <stamp>.event.json     created last, only for a kept clip:
                      the event, in the events index format
                      (pipeline.events.event_entry): event_id <stamp>-<CAMERA>,
                      camera, recorded_at, paths (relative to STORAGE_ROOT) of
                      the processed video and metadata (the video is not
                      copied), "original" / "processed": {"folder",
                      "video_file"} video links, the LLM's summary. Also copied
                      to the chat area as events-inbox/<stamp>-<CAMERA>.event.json.
  5. clean up — removes the inbox file and the clip's work folder.

Ignored clips — skipped by the filter, or judged unimportant at any step
(triage: nothing changes, OpenCV: no motion, YOLO: no required class moving,
LLM: ignore, unless it reports a threat or a person — or a vehicle, on a
person_or_vehicle camera) — get their .metadata.json, with
every processing detail and an "ignored": {"at", "by", "reason"} record,
moved to the ignored area (IGNORED_VIDEOS_*, same folder structure); the cut
video is deleted.

--preserve (testing): nothing is deleted — files are copied (not moved) out
of the work folder, and the work folder and inbox file stay. A failed clip
also keeps its inbox file and work folder, and is retried next run.

A clip that was processed (and kept) before reuses that run's trimmed clip and
triage/OpenCV/YOLO results (YOLO rechecked against YOLO_REQUIRED_CLASSES) and,
if kept with --preserve, its frames — only the LLM runs again. --redo reruns
everything.

Queued events are picked up by the notifier (`python -m blink_watcher notify`).

Usage:
    python -m blink_watcher process                # CAMERA_FILTERS / PROCESS_LIMIT from config
    python -m blink_watcher process --limit 5
    python -m blink_watcher process --filter always --limit 0
    python -m blink_watcher process --limit 5 -v     # + internal module logs
    python -m blink_watcher process --limit 1 --preserve  # testing: keep inbox file + work folder
    python -m blink_watcher process --limit 1 --preserve --redo  # ... and rerun every step
    python -m blink_watcher process --daemon       # keep running, poll the inbox every second
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import PurePosixPath
from typing import Optional

from blink_watcher.core.daemon import run_daemon
from blink_watcher.core import security_mode
from blink_watcher.core.filters import FILTER_NAMES, describe, passes, summary
from blink_watcher.core.logs import setup_logging
from blink_watcher.core.storage import Storage, link_key
from blink_watcher.describe.describer import VideoDescriber
from blink_watcher.pipeline.process.clip import INBOX, Areas, Clip, Options
from blink_watcher.pipeline.process.processor import ClipProcessor
from blink_watcher.settings import Settings, load_settings

log = logging.getLogger("blink_watcher.pipeline.process")

PROGRAM = "process"

Entry = tuple[str, dict]  # (inbox key, queued metadata)


def read_inbox(processed: Storage) -> list[Entry]:
    """Every queued clip, oldest recording first."""
    entries = []
    for key in processed.list(INBOX):
        if not key.endswith(".metadata.json"):
            continue
        try:
            entries.append((key, processed.read_json(key)))
        except ValueError as e:
            log.warning(f"Unreadable inbox file {key}: {e} - left in place")
    return sorted(entries, key=lambda e: e[1].get("recorded_at") or "")


def print_header(settings: Settings, log_file, areas: Areas, opts: Options, waiting: int, total: int) -> None:
    need = opts.required_classes
    log.info("Video processor started (log file: %s)", log_file)
    log.info(f"Inbox: {areas.processed.local_path(INBOX)} - {waiting} clip(s) queued, oldest first")
    filters = f"{opts.filter_override} for every camera (--filter)" if opts.filter_override         else summary(settings.camera_filters)
    log.info(f"Filters: {filters} - clips that don't match their camera's filter are skipped")
    log.info(f"Limit: {opts.limit or 'none'} - {total} clip(s) will be processed")
    indoor = ", ".join(c.strip() for c in settings.indoor_cameras) or "none"
    log.info(f"Security mode: {opts.security_mode().headline()} - read per clip; indoor cameras ({indoor}) "
             f"are saved without analysis in HOME")
    log.info(f"YOLO required classes: {', '.join(need) if need else 'any moving object'}")
    log.info(f"Describer: {settings.describer_model} via {settings.llm_provider}"
             + (f", effort {settings.describer_effort}" if settings.describer_effort else ""))
    log.info(f"Preserve: {'enabled - no files will be deleted' if opts.preserve else 'disabled'}")
    log.info(f"Redo: {'enabled - all steps will be executed' if opts.redo else 'disabled - prior results will be reused'}")
    log.info(f"Daemon: {'enabled' if opts.daemon else 'disabled - exits after this run'}")


def process_batch(processor: ClipProcessor, batch: list[Entry]) -> list[str]:
    """Process `batch`; returns the inbox keys of the clips that failed."""
    areas, opts = processor.areas, processor.opts
    kept = dropped = skipped = 0
    failed: list[str] = []
    t_run = time.perf_counter()
    for n, (inbox_key, md) in enumerate(batch, start=1):
        clip = Clip(inbox_key, md, link_key(areas.raw, md["original"]))
        detections = md["original"].get("provider_detections") or []
        when = (md.get("recorded_at") or "?")[:19].replace("T", " ")
        log.info(f"[{n}/{len(batch)}] Processing {md.get('camera')} {when} ({PurePosixPath(inbox_key).name})")
        log.info(f"  Provider detections: {', '.join(detections) or 'none'}")
        camera_filter = opts.filter_of(md.get("camera") or "")
        if not passes(camera_filter, detections):
            processor.skip_by_filter(clip, detections, camera_filter)
            skipped += 1
            continue
        # "always": the camera has no person detection of its own; triage / YOLO / the LLM still decide.
        log.info(f"  Filter '{camera_filter}' matched"
                 + ("" if camera_filter == "always" else f" ({describe(camera_filter)})"))
        try:
            if processor.process(clip):
                kept += 1
            else:
                dropped += 1
        except Exception as e:
            # Leave the inbox file so the clip is retried next run.
            log.exception("failed to process %s", inbox_key)
            log.error(f"  Processing failed: {type(e).__name__}: {e} - inbox entry retained for retry")
            failed.append(inbox_key)

    left = len(read_inbox(areas.processed))
    log.info(f"Run completed in {time.perf_counter() - t_run:.1f}s: {kept} kept, {dropped} ignored, "
             f"{skipped} skipped by filter, {len(failed)} failed; {left} remaining in inbox")
    return failed


def run(settings: Settings, opts: Options, log_file) -> int:
    areas = Areas.from_settings(settings)
    describer = VideoDescriber.from_settings(
        settings, progress=lambda msg, level=logging.INFO: log.log(level, f"        {msg}"))
    processor = ClipProcessor(areas, opts, describer)
    entries = read_inbox(areas.processed)
    total = min(opts.limit, len(entries)) if opts.limit else len(entries)
    print_header(settings, log_file, areas, opts, len(entries), total)

    def next_batch(held: set[str]) -> list[Entry]:
        due = [e for e in read_inbox(areas.processed) if e[0] not in held]
        return due[:opts.limit] if opts.limit else due

    if opts.daemon:
        return run_daemon(log, "clips", next_batch, lambda batch: process_batch(processor, batch))
    return 1 if process_batch(processor, entries[:total]) else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", choices=FILTER_NAMES, default=None,
                        help="One detection filter for every camera, instead of CAMERA_FILTERS")
    parser.add_argument("--limit", type=int, default=None,
                        help="Override PROCESS_LIMIT (clips per run, 0 = no limit)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Also show the pipeline modules' internal INFO logs")
    parser.add_argument("--preserve", action="store_true",
                        help="Testing: delete nothing - keep the inbox files (clips stay queued) and the "
                             "extracted frames + LLM request/result; copy ignored clips instead of moving")
    parser.add_argument("--redo", action="store_true",
                        help="Rerun every step; by default an already processed clip reuses its checks, "
                             "trimmed clip and preserved frames, and only the LLM runs again")
    parser.add_argument("--daemon", action="store_true",
                        help="Keep running: poll the inbox every second while it is empty (--limit = per batch)")
    args = parser.parse_args(argv)
    if args.daemon and args.preserve:
        parser.error("--daemon cannot be combined with --preserve (the same clips would be processed forever)")
    log_file = setup_logging(PROGRAM, args.verbose, log.name)

    settings = load_settings()
    return run(settings, Options(
        limit=settings.process_limit if args.limit is None else args.limit,
        preserve=args.preserve,
        redo=args.redo,
        required_classes=settings.yolo_required_classes,
        daemon=args.daemon,
        filter_override=args.filter,
        camera_filter=settings.camera_filter,
        security_mode=lambda: security_mode.current(settings),
        indoor_camera=settings.camera_indoor,
    ), log_file)
