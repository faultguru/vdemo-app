"""ClipProcessor: the steps one queued clip goes through (see
blink_watcher.pipeline.process for the whole picture). Everything a clip
produces is written to its work folder in the temp area first and moved out
in the save step."""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

from blink_watcher.chat import store as chat_store
from blink_watcher.core.security_mode import ModeState, prompt_block
from blink_watcher.core.storage import file_link
from blink_watcher.describe.describer import VideoDescriber
from blink_watcher.pipeline.process.clip import Areas, Clip, Options, transfer
from blink_watcher.pipeline.process.record import (
    build_event, build_record, fast_summary, ignore_decision, ignored, llm_block, mark_empty_ignored, shift_segments,
)
from blink_watcher.vision.check import check_video
from blink_watcher.vision.focus import Focus, find_focus, trim_to_focus
from blink_watcher.vision.motion import MotionSegment
from blink_watcher.vision.tracks import PERSON_MOVE_RATIO
from blink_watcher.vision.triage import TRIAGE_CHANGED_RATIO
from blink_watcher.vision.video import encode_jpeg, read_frames_at, video_info
from blink_watcher.vision.yolo import yolo_verdict

log = logging.getLogger("blink_watcher.pipeline.process")

STEP = "        "  # indent of a step's detail lines
FAST_SKIP_REASON = "security mode HOME: indoor clip saved without analysis or notification"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mb(n: int) -> str:
    return f"{n / 2**20:.1f} MB"


def _spans(segments: list[MotionSegment], limit: int = 6) -> str:
    text = ", ".join(f"{s.start_sec:.1f}-{s.end_sec:.1f}s" for s in segments[:limit])
    return text + (f", ... (+{len(segments) - limit} more)" if len(segments) > limit else "")


def report_check(proc: dict, segments: list[MotionSegment]) -> None:
    """One line per triage / OpenCV / YOLO record."""
    sec = "reused" if proc.get("reused") else f"{(proc.get('duration_ms') or 0) / 1000:5.1f}s"
    name = proc.get("name")
    if name == "triage":
        changed = f"{proc.get('changed_ratio', 0):.1%} of pixels changed (threshold {TRIAGE_CHANGED_RATIO:.1%})"
        log.info(f"{STEP}triage  {sec}  " + (f"Frame change detected: {changed}" if proc.get("passed")
                                            else f"No frame change detected: {changed}"))
    elif name == "opencv":
        log.info(f"{STEP}opencv  {sec}  " + (f"{len(segments)} motion segment(s) detected [{_spans(segments)}]"
                                            if segments else "No motion detected"))
    elif name == "yolo":
        need = proc.get("required_classes")
        found = ", ".join(proc.get("classes_detected") or []) or "none"
        if need:
            verdict = f"{' / '.join(need).capitalize()} {'found' if proc.get('passed') else 'not found'}"
            log.info(f"{STEP}yolo    {sec}  {verdict} (moving objects: {found})")
        else:
            log.info(f"{STEP}yolo    {sec}  " + (f"Moving objects found: {found}" if proc.get("passed")
                                                else "No moving objects found"))


@dataclass
class Analysis:
    """The clip on the trimmed video's clock, ready to describe."""
    info: dict
    segments: list[MotionSegment]
    processors: list[dict]          # triage, opencv, yolo, trim
    focus: Optional[Focus] = None   # the YOLO scan, when it ran in this run
    reused: bool = False            # from an earlier run's processed metadata


class ClipProcessor:
    def __init__(self, areas: Areas, opts: Options, describer: VideoDescriber):
        self.areas = areas
        self.opts = opts
        self.describer = describer

    # ---------- One clip ----------

    def process(self, clip: Clip) -> bool:
        """Steps 1-5: True if the clip was kept, False if it was ignored. If a
        step fails, the inbox file and work folder stay (retried next run)."""
        t0 = time.perf_counter()
        kept = self._run(clip)
        self._finish(clip, t0)
        return kept

    def _run(self, clip: Clip) -> bool:
        areas = self.areas
        src = areas.raw.local_path(clip.key)
        orig = clip.md["original"]
        log.info(f"  Source video: {src} ({_mb(areas.raw.size(clip.key))}, {orig.get('duration_sec')}s, "
                 f"provider={orig.get('provider')}, trigger={orig.get('trigger')})")
        house = self.opts.security_mode()
        clip.md["security_mode"] = house.mode  # in every record of the clip, kept or ignored
        if house.mode == "home" and self.opts.indoor_camera(clip.camera):
            return self._run_fast(clip, src)
        log.info(f"  Security mode: {house.headline()}")

        prior = self._load_prior(clip)
        if prior is None:
            areas.temp.delete_tree(clip.work_key)  # fresh run: no stale files from an earlier run
        log.info("  Mode: " + ("redo requested - all steps will be executed" if self.opts.redo else
                               "prior results found - analysis and trim will be reused" if prior else
                               "no prior results - all steps will be executed"))
        log.info(f"  Work folder: {areas.temp.folder}/{clip.work_key}/")
        trimmed = areas.temp.local_path(clip.work_file(".mp4"))
        trimmed.parent.mkdir(parents=True, exist_ok=True)

        if prior:
            analysis = self._reuse(clip, prior)
        else:
            check = self._check(clip, src)
            if check is None:
                return False
            analysis = self._trim(clip, src, trimmed, *check)
        # Where the web console sees that the clip is being described.
        areas.temp.write_json(clip.work_file(".analysis.json"), {
            "info": analysis.info, "processors": analysis.processors,
            "motion_segments": [s.to_record() for s in analysis.segments]})

        processed = file_link(areas.processed.folder, clip.key)
        record = build_record(clip.md, analysis.info, analysis.segments, analysis.processors, processed=processed)
        decision = ignore_decision(record, None)
        if decision:
            log.info("  [3/5] Description skipped - nothing to describe")
            self._save_ignored(clip, record, decision)
            return False

        llm_proc, llm = self._describe(clip, trimmed, analysis, house)
        record = build_record(clip.md, analysis.info, analysis.segments, analysis.processors + [llm_proc], llm,
                              processed=processed)
        decision = ignore_decision(record, llm, keep_vehicles=self.opts.filter_of(clip.camera) == "person_or_vehicle")
        if decision:
            self._save_ignored(clip, record, decision)
            return False
        self._save_kept(clip, record, self._thumbnail(clip, trimmed, analysis.segments, llm))
        return True

    def _run_fast(self, clip: Clip, src: Path) -> bool:
        """Home security mode, indoor camera: no analysis at all. The original
        video is saved as the processed one with the clip's metadata and a
        thumbnail (its middle frame), and the clip goes straight into its day's
        events index (shown and stitched in the console); no check, trim,
        description, event file or notification."""
        areas = self.areas
        log.info("  Security mode: HOME, indoor camera - fast path: saved as it is, "
                 "no check, trim, description or notification")
        log.info("  [1/5] Analysis skipped - security mode HOME")
        log.info("  [2/5] Trimming skipped - original video kept")
        log.info("  [3/5] Description skipped")
        areas.temp.delete_tree(clip.work_key)
        video = areas.temp.local_path(clip.work_file(".mp4"))
        video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, video)
        info = video_info(video)
        record = build_record(clip.md, info, [], [], processed=file_link(areas.processed.folder, clip.key))
        record["fast"] = {"reason": "security mode home, indoor camera", "at": _now()}
        middle = round(float(info.get("duration_sec") or 0.0) / 2, 3)
        thumbnail = self._thumbnail(clip, video, [], {"important_frames_sec": [middle], "thumbnail_sec": middle})
        self._save_fast(clip, record, thumbnail)
        return True

    # ---------- Steps 1 + 2, or an earlier run's ----------

    def _load_prior(self, clip: Clip) -> Optional[dict]:
        """The earlier run's processed metadata for this clip, if it can be reused."""
        processed = self.areas.processed
        if self.opts.redo or not (processed.exists(clip.key) and processed.exists(clip.metadata_key)):
            return None
        prior = processed.read_json(clip.metadata_key)
        return prior if any(p.get("name") == "trim" for p in prior.get("processors") or []) else None

    def _reuse(self, clip: Clip, prior: dict) -> Analysis:
        """Steps 1 + 2 from the earlier run: its trimmed clip and checks, YOLO
        rechecked against YOLO_REQUIRED_CLASSES."""
        log.info("  [1/5] Analysis skipped - prior results reused")
        procs = {p["name"]: {**p, "reused": True} for p in prior["processors"]}
        segments = [MotionSegment.from_record(s) for s in prior.get("motion_segments") or []]
        for name in ("opencv", "yolo"):
            if name in procs:
                if name == "yolo":
                    yolo_verdict(procs[name], self.opts.required_classes)
                report_check(procs[name], segments)
        trim = procs["trim"]
        self.areas.temp.write_bytes(clip.work_file(".mp4"), self.areas.processed.read_bytes(clip.key))
        log.info(f"  [2/5] Trimming skipped - prior trimmed video reused ({trim.get('duration_sec')}s)")
        processors = [procs[n] for n in ("triage", "opencv", "yolo") if n in procs] + [trim]
        return Analysis(prior.get("info") or {}, segments, processors, reused=True)

    def _check(self, clip: Clip, src: Path) -> Optional[tuple[dict, list[MotionSegment], list[dict]]]:
        """Step 1: triage -> OpenCV -> YOLO on the original clip. None when
        the clip is ignored by it (saved as ignored)."""
        log.info("  [1/5] Analyzing original video for motion and persons...")
        result = check_video(src, self.opts.required_classes, progress=report_check)
        record = result.record()
        self.areas.temp.write_json(clip.work_file(".check.json"), record)
        decision = ignore_decision(record, None)
        if decision:
            log.info("  [2/5] Trimming skipped")
            log.info("  [3/5] Description skipped")
            self._save_ignored(clip, build_record(clip.md, result.info, result.segments, result.processors),
                               decision)
            return None
        return result.info, result.segments, result.processors

    def _trim(self, clip: Clip, src: Path, trimmed: Path, info: dict, segments: list[MotionSegment],
              processors: list[dict]) -> Analysis:
        """Step 2: YOLO person scan + cut the quiet head / tail into `trimmed`;
        the check results shifted onto the trimmed clip's clock."""
        log.info("  [2/5] Trimming video to person activity...")
        src_dur = float(info.get("duration_sec") or 0.0)
        t0 = time.perf_counter()
        focus = find_focus(src, info, segments)
        scan_sec = f"{time.perf_counter() - t0:5.1f}s"
        if focus.source == "person":
            log.info(f"{STEP}scan    {scan_sec}  Moving person found at {focus.span[0]:.1f}-{focus.span[1]:.1f}s")
        elif focus.span:
            log.warning(f"{STEP}scan    {scan_sec}  Moving person not found (moved <{PERSON_MOVE_RATIO:.0%} of "
                        f"frame) - motion span used: {focus.span[0]:.1f}-{focus.span[1]:.1f}s")
        else:
            log.warning(f"{STEP}scan    {scan_sec}  No activity found")

        t1 = time.perf_counter()
        if focus.span:
            trim = trim_to_focus(src, trimmed, info, focus)
            # The same scan on the trimmed clip's clock: frame sampling doesn't run it again.
            focus = focus.shifted(trim["offset_sec"], trim["duration_sec"])
            log.info(f"{STEP}trim    {time.perf_counter() - t1:5.1f}s  Video trimmed: {trim['duration_sec']:.1f}s "
                     f"of {src_dur:.1f}s kept ({trim['removed_head_sec']:.1f}s removed from start, "
                     f"{trim['removed_tail_sec']:.1f}s from end), "
                     f"{_mb(src.stat().st_size)} -> {_mb(trimmed.stat().st_size)}")
        else:
            shutil.copyfile(src, trimmed)
            trim = {"focus": None, "offset_sec": 0.0, "duration_sec": src_dur}
            focus = None
            log.info(f"{STEP}trim            Trim skipped - full video kept")
        trim_proc = {"name": "trim", "completed_at": _now(), "duration_ms": int((time.perf_counter() - t0) * 1000),
                     **trim}
        return Analysis(video_info(trimmed), shift_segments(segments, trim["offset_sec"], trim["duration_sec"]),
                        processors + [trim_proc], focus)

    # ---------- Step 3 ----------

    def _describe(self, clip: Clip, trimmed: Path, analysis: Analysis, house: ModeState) -> tuple[dict, dict]:
        """Frames + LLM on the trimmed clip, told the security mode; frames and the
        LLM request / result go to the work folder. Returns (llm processor
        record, llm block)."""
        d = self.describer
        log.info(f"  [3/5] Generating description with {d.model}...")
        t0 = time.perf_counter()
        description = d.describe(trimmed, self.areas.temp.local_path(clip.work_key), analysis.info,
                                 analysis.segments, trim=False, focus=analysis.focus, reuse_frames=analysis.reused,
                                 security_mode=prompt_block(house))
        duration_ms = int((time.perf_counter() - t0) * 1000)
        completed_at = _now()
        if mark_empty_ignored(description):
            log.info(f"{STEP}result          Empty description - marked not important")
        llm = llm_block(description, analysis.segments, completed_at, duration_ms)
        s = llm["summary"]
        log.info(f"{STEP}result          Description: {s.get('description') or '(none)'}")
        log.info(f"{STEP}result          Classified as {'not important' if s.get('ignore') else 'important'}, "
                 f"{len(llm['events'])} event(s)")
        threat = s.get("threat") or {}
        if (threat.get("level") or "none") != "none":
            log.warning(f"{STEP}result          {threat['level'].capitalize()} activity detected "
                        f"[{', '.join(threat.get('signals') or [])}]: {threat.get('evidence') or ''} - clip is kept")
        proc = {"name": "llm", "provider": llm["provider"], "model": llm["model"], "completed_at": completed_at,
                "duration_ms": duration_ms}
        return proc, llm

    def _thumbnail(self, clip: Clip, trimmed: Path, segments: list[MotionSegment], llm: dict) -> bool:
        """<stamp>.jpg in the work folder: the frame at the LLM's thumbnail_sec,
        else the middle of its important frames (or of the segments' middle
        keyframes). False if there is no frame to take."""
        times = sorted({t for t in llm["important_frames_sec"] if t >= 0}) or sorted(
            {s.keyframe_times_sec[len(s.keyframe_times_sec) // 2] for s in segments if s.keyframe_times_sec})
        wanted = llm["thumbnail_sec"]
        frames = read_frames_at(trimmed, times + ([wanted] if wanted is not None else []))
        frame = frames.get(wanted) if wanted is not None else None
        if frame is None:
            readable = [t for t in times if t in frames]
            if not readable:
                return False
            frame = frames[readable[len(readable) // 2]]
        self.areas.temp.write_bytes(clip.work_file(".jpg"), encode_jpeg(frame))
        return True

    # ---------- Step 4 ----------

    def _save_kept(self, clip: Clip, record: dict, thumbnail: bool, **event_extra) -> None:
        """The metadata and event (+ `event_extra` fields) are written to the
        work folder, then cut video, metadata, thumbnail and event are moved to
        the processed area (copied with --preserve), and the event is copied to
        the chat area's events inbox. Ignored clips never get an event."""
        areas, preserve = self.areas, self.opts.preserve
        verb = "copied (--preserve)" if preserve else "moved"
        log.info(f"  [4/5] Saving results to {areas.processed.folder}/{PurePosixPath(clip.key).parent}/...")
        if not thumbnail:
            log.warning(f"{STEP}Thumbnail not found - event created without thumbnail")
        areas.temp.write_json(clip.work_file(".metadata.json"), record)
        areas.temp.write_json(clip.work_file(".event.json"), build_event(areas, clip, record, thumbnail, **event_extra))
        log.info(f"{STEP}Event created: {PurePosixPath(clip.work_file('.event.json')).name}")

        outputs = [("Video", ".mp4", clip.key), ("Metadata", ".metadata.json", clip.metadata_key)]
        if thumbnail:
            outputs.append(("Thumbnail", ".jpg", clip.thumbnail_key))
        else:
            areas.processed.delete(clip.thumbnail_key)  # drop a stale one from an earlier run
        outputs.append(("Event", ".event.json", clip.event_key))  # last: the event only appears once the rest is in place
        for label, suffix, key in outputs:
            transfer(areas.temp, clip.work_file(suffix), areas.processed, key, preserve)
            log.info(f"{STEP}{label} {verb}: {PurePosixPath(key).name}")
        transfer(areas.processed, clip.event_key, areas.chat, clip.event_inbox_key, keep_source=True)
        log.info(f"{STEP}Event queued: {areas.chat.folder}/{clip.event_inbox_key}")
        areas.ignored.delete(clip.metadata_key)  # judged important now; drop an older ignore record

    def _save_fast(self, clip: Clip, record: dict, thumbnail: bool) -> None:
        """A clip saved without analysis: video, metadata and thumbnail moved to
        the processed area (copied with --preserve), and its entry written to the
        day's events index directly — the notifier never sees it."""
        areas, preserve = self.areas, self.opts.preserve
        verb = "copied (--preserve)" if preserve else "moved"
        log.info(f"  [4/5] Saving to {areas.processed.folder}/{PurePosixPath(clip.key).parent}/...")
        areas.temp.write_json(clip.work_file(".metadata.json"), record)
        outputs = [("Video", ".mp4", clip.key), ("Metadata", ".metadata.json", clip.metadata_key)]
        if thumbnail:
            outputs.append(("Thumbnail", ".jpg", clip.thumbnail_key))
        for label, suffix, key in outputs:
            transfer(areas.temp, clip.work_file(suffix), areas.processed, key, preserve)
            log.info(f"{STEP}{label} {verb}: {PurePosixPath(key).name}")
        entry = build_event(areas, clip, record, thumbnail, summary=fast_summary(), fast=record["fast"],
                            notification={"decision": "skip", "skip_reason": FAST_SKIP_REASON})
        path, count = chat_store.add_event(areas.chat.local_path(""), entry)
        log.info(f"{STEP}Added to {areas.chat.folder}/{path.parent.name}/{path.name} ({count} event(s) that day) "
                 f"- no notification")
        areas.processed.delete(clip.event_key)  # from an earlier, analysed run
        areas.chat.delete(clip.event_inbox_key)
        areas.ignored.delete(clip.metadata_key)

    def _save_ignored(self, clip: Clip, record: dict, decision: dict) -> None:
        """Metadata from the work folder to the ignored area (moved, or copied
        with --preserve); the cut video goes with the work folder, and any
        earlier processed copy is removed."""
        areas, preserve = self.areas, self.opts.preserve
        log.info(f"  [4/5] Clip classified as not important by {decision['by']}: {decision['reason']}")
        body = {**{k: v for k, v in record.items() if k != "processed"}, "ignored": decision}
        areas.temp.write_json(clip.work_file(".metadata.json"), body)
        transfer(areas.temp, clip.work_file(".metadata.json"), areas.ignored, clip.metadata_key, preserve)
        log.info(f"{STEP}Metadata {'copied (--preserve)' if preserve else 'moved'} to "
                 f"{areas.ignored.folder}/{clip.metadata_key}")
        if not preserve:
            for key in (clip.key, clip.metadata_key, clip.event_key, clip.thumbnail_key):
                areas.processed.delete(key)
            areas.chat.delete(clip.event_inbox_key)  # an event from an earlier run, not yet picked up
            if "processed" in record:  # a trimmed video was made (not when step 1 stopped the clip)
                log.info(f"{STEP}Trimmed video discarded")

    # ---------- Step 5 ----------

    def _finish(self, clip: Clip, t0: float) -> None:
        """Remove the inbox file and the clip's work folder (both kept with --preserve)."""
        areas = self.areas
        name = PurePosixPath(clip.inbox_key).name
        if self.opts.preserve:
            log.info(f"  [5/5] Cleanup skipped (--preserve): inbox/{name} and "
                     f"{areas.temp.folder}/{clip.work_key}/ retained")
        else:
            areas.processed.delete(clip.inbox_key)
            areas.temp.delete_tree(clip.work_key)
            log.info(f"  [5/5] Cleanup completed: inbox/{name} and {areas.temp.folder}/{clip.work_key}/ removed")
        log.info(f"  Clip processed in {time.perf_counter() - t0:.1f}s")

    # ---------- Filter ----------

    def skip_by_filter(self, clip: Clip, detections: list[str], filter_name: str) -> None:
        """A clip whose provider detections don't match the filter: metadata
        written to the work folder, then moved to the ignored area (copied with
        --preserve, which also keeps the work folder and inbox file); the raw
        clip is untouched."""
        areas, opts = self.areas, self.opts
        reason = f"provider detections [{', '.join(detections) or 'none'}] don't match filter '{filter_name}'"
        areas.temp.write_json(clip.work_file(".metadata.json"), {**clip.md, "ignored": ignored("filter", reason)})
        transfer(areas.temp, clip.work_file(".metadata.json"), areas.ignored, clip.metadata_key, opts.preserve)
        if not opts.preserve:
            areas.processed.delete(clip.inbox_key)
            areas.temp.delete_tree(clip.work_key)
        log.info(f"  Clip skipped: filter '{filter_name}' not matched; metadata "
                 f"{'copied (--preserve, inbox entry retained)' if opts.preserve else 'moved'} to "
                 f"{areas.ignored.folder}/{clip.metadata_key}")
