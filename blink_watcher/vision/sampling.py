"""Dense, action-aware frame sampling for an image-only describer: the LLM gets
still frames, not video, so the frames are chosen here.

  0. Trim (optional): the clip is cut to its focus (focus.py: while people
     move on screen, else the motion span) into <video stem>.trimmed.mp4 in
     the output folder; the original is only read. Every later step runs on
     the trimmed file, so all timestamps are on its timeline;
     manifest["trim"]["offset_sec"] maps them back (original t = t + offset).
  1. The span: the focus, padded by SPAN_PAD_SEC.
  2. Baseline: evenly spaced frames across the span at BASELINE_FPS.
  3. Bursts: around each moment a tracked person's box gets suddenly
     squatter (bending, crouching - the drop-off / pick-up moment), at the
     start and end of each bend, and around each track's first / last
     sighting (before / after hand state): BURST_FPS frames for
     +-BURST_HALF_SEC.
  4. Total frames capped at MAX_FRAMES; the baseline is thinned first.
  5. Every frame is cropped to one region of interest per clip - the union
     of every person box (moving vehicles / animals only when no person is
     seen), padded for context - at the scale an uncropped frame would get,
     so the crop cuts pixels (and image tokens) without losing detail.

Output folder (default <video dir>/<video stem>.details/): the trimmed video,
the JPEGs and manifest.json. Nothing is sent to any model.

Usage:
    python -m blink_watcher frames path/to/clip.mp4
    python -m blink_watcher frames path/to/clip.mp4 --no-trim  # sample the original
    python -m blink_watcher frames path/to/clip.mp4 -o out/dir --max-frames 32
    python -m blink_watcher frames path/to/clip.mp4 --no-roi   # keep full frames
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from blink_watcher.core.logs import setup_logging
from blink_watcher.vision.focus import Focus, find_focus, trim_to_focus
from blink_watcher.vision.motion import MotionSegment, detect_segments
from blink_watcher.vision.tracks import (
    OBJECT_MOVE_RATIO, SCAN_CLASSES, SCAN_FPS, Box, Track, build_tracks, moving_person_tracks,
)
from blink_watcher.vision.video import encode_jpeg, read_frames, video_info, video_ref

log = logging.getLogger(__name__)

PROGRAM = "frames"
MANIFEST = "manifest.json"

# Sampling.
SPAN_PAD_SEC = 0.5             # extend the focus span this much on both sides
BASELINE_FPS = 1.0             # evenly spaced context frames across the whole span
BURST_FPS = 4.0                # frame rate inside a burst
BURST_HALF_SEC = 0.75          # burst covers trigger +- this many seconds
BASELINE_NEAR_BURST_SEC = 0.15 # drop a baseline frame this close to a burst frame
MAX_FRAMES = 48                # cap on frames per clip
FRAME_MAX_SIDE = 960           # JPEG long side of an uncropped frame - ~700 image tokens at 16:9
TOKENS_PER_PIXEL = 1 / 750     # rough image-token cost: width x height / 750

# Region of interest.
ROI_PAD_RATIO = 0.20           # pad the ROI by this fraction of its own size per side - keeps the porch/ground around the subject
ROI_MIN_PAD_RATIO = 0.08       # ...but at least this fraction of the frame's long side, so the scene stays recognizable
ROI_MIN_SAVING = 0.10          # crop only if it removes at least 10% of the frame area

# Bend detection.
STANDING_PERCENTILE = 80       # a track's "standing" box h/w = this percentile over the whole track
BEND_ASPECT_RATIO = 0.70       # box h/w below this x standing h/w -> bent/crouched (h/w is distance-invariant, raw height is not)
TOP_MARGIN_RATIO = 0.02        # boxes touching the top edge have the head cut off - false "bend", skip
EDGE_GUARD_SEC = 1.5           # near a track's start/end, a box touching any edge is a person entering/leaving, not bending - skip
BEND_MERGE_GAP_SEC = 0.4       # bend samples closer than this belong to the same bend; wider would fuse distinct bends (set-down -> pick-up 1s later)
TRIGGER_MERGE_SEC = 1.0        # merge triggers closer than this


@dataclass
class Trigger:
    t: float
    reason: str                    # "bend-start" | "bend-end" | "bend-peak" | "enter" | "exit"
    track: int
    score: Optional[float] = None  # bend depth: box h/w / standing h/w; lower = deeper


def _aspect(box: Box) -> float:
    return (box[3] - box[1]) / max(1.0, box[2] - box[0])


def _touches_edge(box: Box, frame_w: int, frame_h: int) -> bool:
    mx, my = frame_w * TOP_MARGIN_RATIO, frame_h * TOP_MARGIN_RATIO
    return box[0] <= mx or box[1] <= my or box[2] >= frame_w - mx or box[3] >= frame_h - my


def find_triggers(tracks: list[Track], frame_w: int, frame_h: int) -> tuple[list[Trigger], list[Trigger]]:
    """(burst triggers, bend peaks). Bursts go at each track's enter / exit and
    at the start and end of every bend - the transitions show the before /
    after hand state; a long crouch's middle is covered by the baseline. The
    bend peaks (deepest point of each bend) are only recorded in the manifest."""
    triggers: list[Trigger] = []
    peaks: list[Trigger] = []
    top_margin = frame_h * TOP_MARGIN_RATIO
    for tr in tracks:
        triggers.append(Trigger(t=tr.first_t, reason="enter", track=tr.id))
        triggers.append(Trigger(t=tr.last_t, reason="exit", track=tr.id))

        standing = float(np.percentile([_aspect(b) for _, b, _ in tr.hits], STANDING_PERCENTILE))
        bend: list[tuple[float, float]] = []  # (t, score)

        def close_bend() -> None:
            if not bend:
                return
            triggers.append(Trigger(t=bend[0][0], reason="bend-start", track=tr.id))
            triggers.append(Trigger(t=bend[-1][0], reason="bend-end", track=tr.id))
            t, score = min(bend, key=lambda e: e[1])
            peaks.append(Trigger(t=t, reason="bend-peak", track=tr.id, score=round(score, 3)))
            bend.clear()

        for t, box, _conf in tr.hits:
            if box[1] <= top_margin:
                continue
            near_track_edge = t - tr.first_t < EDGE_GUARD_SEC or tr.last_t - t < EDGE_GUARD_SEC
            if near_track_edge and _touches_edge(box, frame_w, frame_h):
                continue
            score = _aspect(box) / standing if standing else 1.0
            if score < BEND_ASPECT_RATIO:
                if bend and t - bend[-1][0] > BEND_MERGE_GAP_SEC:
                    close_bend()
                bend.append((t, score))
        close_bend()

    # Merge near-duplicate triggers, preferring bend transitions over enter/exit.
    triggers.sort(key=lambda g: 0 if g.reason.startswith("bend") else 1)
    kept: list[Trigger] = []
    for g in triggers:
        if not any(abs(g.t - k.t) < TRIGGER_MERGE_SEC for k in kept):
            kept.append(g)
    return sorted(kept, key=lambda g: g.t), sorted(peaks, key=lambda g: g.t)


def _times(start: float, end: float, rate: float) -> list[float]:
    if end < start:
        return []
    n = int((end - start) * rate) + 1
    return [start + k / rate for k in range(n)]


def _thin(items: list, keep: int) -> list:
    """Evenly subsample to `keep` items, always keeping first and last."""
    if keep >= len(items):
        return items
    if keep <= 1:
        return items[:keep]
    step = (len(items) - 1) / (keep - 1)
    return [items[int(round(k * step))] for k in range(keep)]


def plan_frames(span: tuple[float, float], triggers: list[Trigger], fps: float, last_frame: int,
                max_frames: int) -> list[dict]:
    """[{frame_idx, t, kind, reasons}] sorted by time; kind "burst" | "baseline"."""
    def to_idx(t: float) -> int:
        return min(max(int(round(t * fps)), 0), last_frame)

    burst: dict[int, set[str]] = {}
    centres: set[int] = set()
    for g in triggers:
        centres.add(to_idx(g.t))
        for t in _times(g.t - BURST_HALF_SEC, g.t + BURST_HALF_SEC, BURST_FPS):
            burst.setdefault(to_idx(t), set()).add(f"{g.reason}:p{g.track}")

    # A third of the budget is reserved for the baseline, so the timeline has no holes.
    baseline_all = sorted({to_idx(t) for t in _times(span[0], span[1], BASELINE_FPS)})
    reserve = min(len(baseline_all), max(2, max_frames // 3))
    budget = max_frames - reserve
    burst_idx = sorted(burst)
    if len(burst_idx) > budget:
        # Too many bursts: keep every trigger's centre frame, thin the rest.
        others = [i for i in burst_idx if i not in centres]
        burst_idx = sorted(centres | set(_thin(others, max(0, budget - len(centres)))))

    near = BASELINE_NEAR_BURST_SEC * fps
    baseline = [i for i in baseline_all if not any(abs(i - b) <= near for b in burst_idx)]
    baseline = _thin(baseline, max(2, max_frames - len(burst_idx)))

    plan = [{"frame_idx": i, "kind": "burst", "reasons": sorted(burst[i])} for i in burst_idx]
    plan += [{"frame_idx": i, "kind": "baseline", "reasons": []} for i in baseline]
    for p in plan:
        p["t"] = round(p["frame_idx"] / fps, 3)
    return sorted(plan, key=lambda p: p["frame_idx"])


def _segment_label(t: float, segments: list[MotionSegment]) -> str:
    for seg in segments:
        if seg.start_sec <= t <= seg.end_sec:
            return str(seg.index)
    return "gap"


def compute_roi(person_tracks: list[Track], other_tracks: list[Track], segments: list[MotionSegment],
                frame_w: int, frame_h: int) -> tuple[Optional[list[int]], str]:
    """([x0, y0, x1, y1], source) or (None, reason) when cropping isn't worth it.
    The union of every person box and every moving vehicle / animal box; the
    OpenCV motion bboxes when YOLO saw nothing."""
    long_side = max(frame_w, frame_h)
    boxes = [b for tr in person_tracks for _, b, _ in tr.hits]
    boxes += [b for tr in other_tracks if tr.moved(long_side * OBJECT_MOVE_RATIO) for _, b, _ in tr.hits]
    source = "yolo"
    if not boxes:
        boxes = [[s.bbox[0], s.bbox[1], s.bbox[0] + s.bbox[2], s.bbox[1] + s.bbox[3]] for s in segments]
        source = "motion"
    if not boxes:
        return None, "nothing detected"

    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    min_pad = long_side * ROI_MIN_PAD_RATIO
    pad_x = max(min_pad, (x1 - x0) * ROI_PAD_RATIO)
    pad_y = max(min_pad, (y1 - y0) * ROI_PAD_RATIO)
    roi = [
        int(max(0, x0 - pad_x)), int(max(0, y0 - pad_y)),
        int(min(frame_w, x1 + pad_x)), int(min(frame_h, y1 + pad_y)),
    ]
    area = (roi[2] - roi[0]) * (roi[3] - roi[1]) / float(frame_w * frame_h)
    if area > 1.0 - ROI_MIN_SAVING:
        return None, f"saving too small (roi covers {area:.0%} of frame)"
    return roi, source


def _shift_segments(segments: list[MotionSegment], offset: float, duration: float) -> list[MotionSegment]:
    """Copies on a cut's clock; segments entirely outside it are dropped. Indices stay the originals."""
    return [
        replace(
            s,
            start_sec=round(max(0.0, s.start_sec - offset), 3),
            end_sec=round(min(duration, s.end_sec - offset), 3),
            keyframe_times_sec=[round(t - offset, 3) for t in s.keyframe_times_sec if 0 <= t - offset <= duration],
        )
        for s in segments
        if s.end_sec - offset > 0 and s.start_sec - offset < duration
    ]


def sample_frames(
    video_path: Path,
    out_dir: Path,
    max_frames: int = MAX_FRAMES,
    roi: bool = True,
    info: Optional[dict] = None,
    segments: Optional[list[MotionSegment]] = None,
    trim: bool = True,
    focus: Optional[Focus] = None,
) -> Path:
    """Sample `video_path` into `out_dir` (JPEGs + manifest.json); returns the
    manifest path. `info` + `segments` from an earlier motion pass skip a
    second one; `focus` (on `video_path`'s clock) skips the YOLO scan. With
    `trim`, the clip is first cut to its focus (<video stem>.trimmed.mp4)."""
    if info is None or segments is None:
        info, segments = detect_segments(video_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    source_info = info
    focus = focus or find_focus(video_path, info, segments)

    work_path = video_path
    offset = 0.0
    trim_meta = None
    if focus.span and trim:
        work_path = out_dir / f"{video_path.stem}.trimmed.mp4"
        trim_meta = trim_to_focus(video_path, work_path, info, focus)
        info = video_info(work_path)
        offset = trim_meta["offset_sec"]

    # Everything below is on the work video's timeline (trimmed when trimming).
    fps = info.get("fps") or 25.0
    duration = float(info.get("duration_sec") or 0.0)
    width, height = info.get("width") or 0, info.get("height") or 0
    last_frame = max(0, int(info.get("frame_count") or round(duration * fps)) - 1)
    segments = _shift_segments(segments, offset, duration)
    focus = focus.shifted(offset, duration)
    log.info("Video: %s - %d motion segment(s) in range", info, len(segments))

    manifest: dict = {
        "video": video_ref(work_path),
        "info": info,
        "source_video": video_ref(video_path),
        "source_info": source_info,
        "trim": trim_meta,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {
            "baseline_fps": BASELINE_FPS, "scan_fps": SCAN_FPS, "burst_fps": BURST_FPS,
            "burst_half_sec": BURST_HALF_SEC, "max_frames": max_frames,
            "frame_max_side": FRAME_MAX_SIDE,
        },
        "segments": [
            {"index": s.index, "start_sec": s.start_sec, "end_sec": s.end_sec, "bbox": list(s.bbox)}
            for s in segments
        ],
        "focus": None,
        "span": None,
        "roi": None,
        "person_tracks": [],
        "object_tracks": [],
        "triggers": [],
        "frames": [],
        "totals": {"frames": 0, "approx_image_tokens": 0},
    }

    if focus.span:
        f0, f1 = focus.span
        manifest["focus"] = {"source": focus.source, "start_sec": f0, "end_sec": f1}
        span = (max(0.0, f0 - SPAN_PAD_SEC), min(duration, f1 + SPAN_PAD_SEC))
        manifest["span"] = {"start_sec": round(span[0], 3), "end_sec": round(span[1], 3)}

        people = moving_person_tracks(focus.samples, width, height)
        others = {name: build_tracks(focus.samples, {name}) for name in SCAN_CLASSES if name != "person"}
        triggers, bend_peaks = find_triggers(people, width, height)
        log.info("Person scan: %d samples, %d track(s), %d trigger(s), %d bend(s)",
                 len(focus.samples), len(people), len(triggers), len(bend_peaks))
        manifest["person_tracks"] = [
            {"id": tr.id, "first_sec": tr.first_t, "last_sec": tr.last_t, "samples": len(tr.hits)}
            for tr in people
        ]
        long_side = max(width, height)
        manifest["object_tracks"] = [
            {"class": name, "id": tr.id, "first_sec": tr.first_t, "last_sec": tr.last_t,
             "samples": len(tr.hits), "moved": tr.moved(long_side * OBJECT_MOVE_RATIO)}
            for name, trs in others.items() for tr in trs
        ]

        # With people present the crop follows them only; vehicles / animals
        # widen it only in person-less clips.
        roi_others = [] if people else [tr for trs in others.values() for tr in trs]
        roi_box, roi_source = compute_roi(people, roi_others, segments, width, height) if roi else (None, "disabled")
        if roi_box:
            area = (roi_box[2] - roi_box[0]) * (roi_box[3] - roi_box[1]) / float(width * height)
            manifest["roi"] = {"box": roi_box, "source": roi_source, "area_ratio": round(area, 3)}
            log.info("ROI: %s from %s (%.0f%% of frame)", roi_box, roi_source, area * 100)
        else:
            log.info("ROI: none (%s) - writing full frames", roi_source)
        # The scale an uncropped frame would get, so cropping cuts pixels, not detail.
        frame_scale = min(1.0, FRAME_MAX_SIDE / max(1, long_side))
        manifest["triggers"] = [
            {"t": g.t, "reason": g.reason, "track": g.track, "score": g.score}
            for g in sorted(triggers + bend_peaks, key=lambda g: g.t)
        ]

        plan = plan_frames(span, triggers, fps, last_frame, max_frames)
        decoded = read_frames(work_path, [p["frame_idx"] for p in plan])
        for p in plan:
            frame = decoded.get(p["frame_idx"])
            if frame is None:
                log.warning("Could not decode frame at %.2fs", p["t"])
                continue
            if roi_box:
                frame = frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]
            h, w = frame.shape[:2]
            max_side = int(round(max(h, w) * frame_scale))
            name = f"{p['t']:07.2f}s_{p['kind']}.jpg"
            (out_dir / name).write_bytes(encode_jpeg(frame, max_side=max_side))
            scale = min(1.0, max_side / max(h, w))
            w, h = int(w * scale), int(h * scale)
            manifest["frames"].append({
                "file": name, "t": p["t"], "kind": p["kind"], "reasons": p["reasons"],
                "segment": _segment_label(p["t"], segments), "width": w, "height": h,
                "approx_image_tokens": int(w * h * TOKENS_PER_PIXEL),
            })
    else:
        log.info("No motion segments - nothing to sample")

    frames = manifest["frames"]
    manifest["totals"] = {"frames": len(frames), "approx_image_tokens": sum(f["approx_image_tokens"] for f in frames)}
    out = out_dir / MANIFEST
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Wrote %d frame(s) (~%d image tokens) -> %s",
             manifest["totals"]["frames"], manifest["totals"]["approx_image_tokens"], out_dir)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("-o", "--out-dir", type=Path, default=None,
                        help="Output directory (default: <video dir>/<video stem>.details/)")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="Cap on frames per clip")
    parser.add_argument("--no-roi", action="store_true", help="Write uncropped full frames")
    parser.add_argument("--no-trim", action="store_true", help="Sample the original video instead of a trimmed copy")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also show other modules' INFO logs")
    args = parser.parse_args(argv)
    setup_logging(PROGRAM, args.verbose, "blink_watcher.vision")

    if not args.video.exists():
        log.error("Video not found: %s", args.video)
        return 1
    out_dir = args.out_dir or args.video.parent / f"{args.video.stem}.details"
    sample_frames(args.video, out_dir, max_frames=args.max_frames, roi=not args.no_roi, trim=not args.no_trim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
