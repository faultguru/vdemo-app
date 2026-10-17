"""Dense, action-aware frame sampling for image-only describers (Claude).

Claude takes images, not video, so we choose the frames ourselves instead of
the fixed 8-per-segment keyframes describe_video uses:

  0. Trim to people: YOLO scans the whole clip; the lead-in before the first
     person sighting and the tail after the last are cut (keeping TRIM_PAD_SEC
     of each) — a car passing before anyone arrives is dropped. Nothing in
     between is removed. Clips with no person fall back to the OpenCV motion
     segments (same detector as describe_video). Written with ffmpeg (video re-encoded for exact
     cuts, audio copied) as <video>.trimmed.mp4 in the output dir — the
     original is only read. Every later step runs on the trimmed file, so all
     timestamps are on the trimmed timeline; manifest["trim"]["offset_sec"]
     maps them back (original t = trimmed t + offset).
  1. The span: first → last person sighting (or motion, without people).
  2. Baseline: evenly spaced full frames across the span at BASELINE_FPS.
  3. Bursts: YOLO tracks people at SCAN_FPS. Around each moment a person's
     box gets suddenly shorter/squatter (bending, crouching — the drop-off /
     pick-up moment) — at the start and end of each bend — and around each
     track's first/last sighting (before / after hand state) we add
     BURST_FPS frames for ±BURST_HALF_SEC.
  4. (disabled) A high-res person crop at each bend's deepest point.
  5. Total full frames capped at MAX_FRAMES; baseline is thinned first.
  6. Every frame is cropped to one fixed region of interest per clip - the
     union of every person box (moving vehicles/animals only when no person
     is seen), padded for surrounding context - at the same scale an
     uncropped frame would get, so the crop cuts pixels (and image tokens)
     without changing detail.

Output dir (default <video dir>/<video stem>.details/): the trimmed video,
the JPEGs and manifest.json. Nothing is sent to any model.

Usage:
    python -m blink_watcher.frame_sampler path/to/clip.mp4
    python -m blink_watcher.frame_sampler path/to/clip.mp4 --no-trim  # sample the original
    python -m blink_watcher.frame_sampler path/to/clip.mp4 -o out/dir --max-frames 32
    python -m blink_watcher.frame_sampler path/to/clip.mp4 --no-roi   # keep full frames
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from blink_watcher.describe_video import (
    _get_yolo,
    _read_frames_at,
    _video_ref,
    detect_segments,
)
from blink_watcher.describers import _jpeg_bytes

log = logging.getLogger("blink_watcher.frame_sampler")

# Trim knobs.
TRIM_PAD_SEC = 1.0             # keep this much quiet video before the first and after the last motion
TRIM_CRF = 23                  # x264 quality for the trimmed video (lower = better)
TRIM_PRESET = "veryfast"       # x264 speed/size trade-off
TRIM_MAX_RATE_RATIO = 0.90     # cap the re-encode at 90% of the source's average bitrate, so the trimmed file is always smaller

# Sampling knobs.
SPAN_PAD_SEC = 0.5             # extend the motion span this much on both sides
BASELINE_FPS = 1.0             # evenly spaced context frames across the whole span
SCAN_FPS = 5.0                 # YOLO person-scan rate used to find bursts
BURST_FPS = 4.0                # frame rate inside a burst
BURST_HALF_SEC = 0.75          # burst covers trigger ± this many seconds
BASELINE_NEAR_BURST_SEC = 0.15 # drop a baseline frame this close to a burst frame
MAX_FRAMES = 48                # cap on frames per clip
FRAME_MAX_SIDE = 960           # full-frame JPEG long side — ~700 image tokens at 16:9
# Person close-up crops disabled - every frame gets the activity-region crop instead.
# CROP_MAX_SIDE = 1024           # person-crop JPEG long side
# CROP_PAD_RATIO = 0.25          # pad person box by this fraction before cropping

# Region-of-interest crop knobs (full frames).
ROI_PAD_RATIO = 0.20           # pad the ROI by this fraction of its own size per side - keeps the porch/ground around the subject
ROI_MIN_PAD_RATIO = 0.08       # ...but at least this fraction of the frame's long side, so the scene stays recognizable
ROI_MIN_SAVING = 0.10          # crop only if it removes at least 10% of the frame area
ROI_MOVE_RATIO = 0.03          # a non-person track counts only if its center moves >=3% of the frame's long side (parked cars don't)

# Detection / person tracking / bend detection knobs.
SCAN_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat")
DETECT_CONF = 0.35             # min YOLO confidence for a box
YOLO_BATCH = 16                # frames per YOLO predict call
TRACK_IOU = 0.3                # min IoU to continue a track
TRACK_MAX_GAP_SEC = 1.0        # a track ends if unseen this long
MIN_TRACK_SAMPLES = 3          # ignore tracks shorter than this (flicker)
STANDING_PERCENTILE = 80       # a track's "standing" box h/w = this percentile over the whole track
BEND_ASPECT_RATIO = 0.70       # box h/w below this × standing h/w → bent/crouched (h/w is distance-invariant, raw height is not)
TOP_MARGIN_RATIO = 0.02        # boxes touching the top edge have the head cut off — false "bend", skip
EDGE_GUARD_SEC = 1.5           # near a track's start/end, a box touching any edge is a person entering/leaving, not bending — skip
BEND_MERGE_GAP_SEC = 0.4       # bend samples closer than this belong to the same bend; wider would fuse distinct bends (set-down → pick-up 1s later)
# CROP_MERGE_SEC = 1.5           # bend-peak crops closer than this are one bend split by a half-rise — keep the deeper
TRIGGER_MERGE_SEC = 1.0        # merge triggers closer than this

# Rough Claude image-token cost: width × height / 750.
TOKENS_PER_PIXEL = 1 / 750


@dataclass
class _Track:
    id: int
    hits: list[tuple[float, list[float], float]] = field(default_factory=list)  # (t, xyxy, conf)

    @property
    def last_t(self) -> float:
        return self.hits[-1][0]

    @property
    def last_box(self) -> list[float]:
        return self.hits[-1][1]


@dataclass
class _Trigger:
    t: float
    reason: str                    # "bend-start" | "bend-end" | "bend-peak" | "enter" | "exit"
    track: int
    box: Optional[list[float]] = None
    score: Optional[float] = None  # bend depth: box h/w ÷ standing h/w; lower = deeper


def _iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _aspect(box: list[float]) -> float:
    return (box[3] - box[1]) / max(1.0, box[2] - box[0])


def scan_objects(
    video_path: Path, fps: float, start_sec: float, end_sec: float
) -> list[tuple[float, list[tuple[list[float], float, str]]]]:
    """Decode [start, end] sequentially, run YOLO (SCAN_CLASSES only) every
    1/SCAN_FPS seconds. Returns [(t, [(xyxy, conf, class_name), ...]), ...]."""
    model = _get_yolo()
    class_ids = [i for i, n in model.names.items() if n in SCAN_CLASSES]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    start_f = max(0, int(round(start_sec * fps)))
    end_f = int(round(end_sec * fps))
    step = max(1, int(round(fps / SCAN_FPS)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    samples: list[tuple[float, list[tuple[list[float], float, str]]]] = []
    batch_t: list[float] = []
    batch_f: list[np.ndarray] = []

    def _flush() -> None:
        if not batch_f:
            return
        results = model.predict(batch_f, conf=DETECT_CONF, classes=class_ids, verbose=False)
        for t, r in zip(batch_t, results):
            dets = []
            if r.boxes is not None:
                for box in r.boxes:
                    dets.append((
                        [float(v) for v in box.xyxy[0].tolist()],
                        float(box.conf.item()),
                        model.names[int(box.cls.item())],
                    ))
            samples.append((t, dets))
        batch_t.clear()
        batch_f.clear()

    for idx in range(start_f, end_f + 1):
        if (idx - start_f) % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            batch_t.append(round(idx / fps, 3))
            batch_f.append(frame)
            if len(batch_f) >= YOLO_BATCH:
                _flush()
        elif not cap.grab():
            break
    _flush()
    cap.release()
    return samples


def build_tracks(
    samples: list[tuple[float, list[tuple[list[float], float, str]]]], classes: set[str]
) -> list[_Track]:
    """Greedy IoU tracker over detections of the given classes — good enough
    to tell 2–3 people on a porch apart."""
    tracks: list[_Track] = []
    for t, all_dets in samples:
        dets = [(box, conf) for box, conf, name in all_dets if name in classes]
        matched: set[int] = set()
        for box, conf in sorted(dets, key=lambda d: -d[1]):
            best, best_iou = None, TRACK_IOU
            for tr in tracks:
                if tr.id in matched or t - tr.last_t > TRACK_MAX_GAP_SEC:
                    continue
                iou = _iou(box, tr.last_box)
                if iou >= best_iou:
                    best, best_iou = tr, iou
            if best is None:
                best = _Track(id=len(tracks))
                tracks.append(best)
            best.hits.append((t, box, conf))
            matched.add(best.id)
    return [tr for tr in tracks if len(tr.hits) >= MIN_TRACK_SAMPLES]


def _touches_edge(box: list[float], frame_w: int, frame_h: int) -> bool:
    mx, my = frame_w * TOP_MARGIN_RATIO, frame_h * TOP_MARGIN_RATIO
    return box[0] <= mx or box[1] <= my or box[2] >= frame_w - mx or box[3] >= frame_h - my


def find_triggers(tracks: list[_Track], frame_w: int, frame_h: int) -> tuple[list[_Trigger], list[_Trigger]]:
    """Return (burst_triggers, crop_points).

    Bursts go at each track's enter/exit and at the start and end of every
    bend episode — the transitions are where before/after hand state is
    visible; a long crouch's middle is covered by the baseline. Crops go at
    each episode's deepest point."""
    triggers: list[_Trigger] = []
    crops: list[_Trigger] = []
    top_margin = frame_h * TOP_MARGIN_RATIO
    for tr in tracks:
        triggers.append(_Trigger(t=tr.hits[0][0], reason="enter", track=tr.id, box=tr.hits[0][1]))
        triggers.append(_Trigger(t=tr.hits[-1][0], reason="exit", track=tr.id, box=tr.hits[-1][1]))

        standing = float(np.percentile([_aspect(b) for _, b, _ in tr.hits], STANDING_PERCENTILE))
        episode: list[tuple[float, list[float], float]] = []  # (t, box, score)

        def _close_episode() -> None:
            if not episode:
                return
            triggers.append(_Trigger(t=episode[0][0], reason="bend-start", track=tr.id, box=episode[0][1]))
            triggers.append(_Trigger(t=episode[-1][0], reason="bend-end", track=tr.id, box=episode[-1][1]))
            t, box, score = min(episode, key=lambda e: e[2])
            crops.append(_Trigger(t=t, reason="bend-peak", track=tr.id, box=box, score=round(score, 3)))
            episode.clear()

        first_t, last_t = tr.hits[0][0], tr.hits[-1][0]
        for t, box, _conf in tr.hits:
            if box[1] <= top_margin:
                continue
            near_track_edge = t - first_t < EDGE_GUARD_SEC or last_t - t < EDGE_GUARD_SEC
            if near_track_edge and _touches_edge(box, frame_w, frame_h):
                continue
            score = _aspect(box) / standing if standing else 1.0
            if score < BEND_ASPECT_RATIO:
                if episode and t - episode[-1][0] > BEND_MERGE_GAP_SEC:
                    _close_episode()
                episode.append((t, box, score))
        _close_episode()

    # Merge near-duplicate triggers, preferring bend transitions over enter/exit.
    triggers.sort(key=lambda g: 0 if g.reason.startswith("bend") else 1)
    kept: list[_Trigger] = []
    for g in triggers:
        if any(abs(g.t - k.t) < TRIGGER_MERGE_SEC for k in kept):
            continue
        kept.append(g)

    # Person close-up crops disabled - every frame gets the activity-region crop instead.
    # # One crop per bend: drop a crop near a deeper one from the same person.
    # kept_crops: list[_Trigger] = []
    # for c in sorted(crops, key=lambda c: c.score):
    #     if any(c.track == k.track and abs(c.t - k.t) < CROP_MERGE_SEC for k in kept_crops):
    #         continue
    #     kept_crops.append(c)
    # Bend peaks are still returned so the manifest records them.
    return sorted(kept, key=lambda g: g.t), sorted(crops, key=lambda c: c.t)


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


def plan_frames(
    span: tuple[float, float],
    triggers: list[_Trigger],
    fps: float,
    last_frame: int,
    max_frames: int,
) -> list[dict]:
    """Return [{frame_idx, t, kind, reasons}] for full frames, sorted by time."""
    to_idx = lambda t: min(max(int(round(t * fps)), 0), last_frame)

    burst: dict[int, set[str]] = {}
    centres: set[int] = set()
    for g in triggers:
        centres.add(to_idx(g.t))
        for t in _times(g.t - BURST_HALF_SEC, g.t + BURST_HALF_SEC, BURST_FPS):
            burst.setdefault(to_idx(t), set()).add(f"{g.reason}:p{g.track}")

    # Reserve a third of the budget for baseline so the timeline has no holes.
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


def _segment_label(t: float, segments) -> str:
    for seg in segments:
        if seg.start_sec <= t <= seg.end_sec:
            return str(seg.index)
    return "gap"


# Person close-up crops disabled - every frame gets the activity-region crop instead.
# def _crop(frame: np.ndarray, box: list[float]) -> np.ndarray:
#     fh, fw = frame.shape[:2]
#     bw, bh = box[2] - box[0], box[3] - box[1]
#     x0 = int(max(0, box[0] - bw * CROP_PAD_RATIO))
#     y0 = int(max(0, box[1] - bh * CROP_PAD_RATIO))
#     x1 = int(min(fw, box[2] + bw * CROP_PAD_RATIO))
#     y1 = int(min(fh, box[3] + bh * CROP_PAD_RATIO))
#     return frame[y0:y1, x0:x1]


def _track_moved(tr: _Track, min_shift: float) -> bool:
    cxs = [(b[0] + b[2]) / 2 for _, b, _ in tr.hits]
    cys = [(b[1] + b[3]) / 2 for _, b, _ in tr.hits]
    return ((max(cxs) - min(cxs)) ** 2 + (max(cys) - min(cys)) ** 2) ** 0.5 >= min_shift


def compute_roi(
    person_tracks: list[_Track], other_tracks: list[_Track], segments, frame_w: int, frame_h: int
) -> tuple[Optional[list[int]], str]:
    """Return ([x0, y0, x1, y1], source) or (None, reason) when cropping
    isn't worth it. Union of every person box plus every moving vehicle/animal
    box; falls back to the OpenCV motion bbox when YOLO saw nothing."""
    long_side = max(frame_w, frame_h)
    boxes = [b for tr in person_tracks for _, b, _ in tr.hits]
    boxes += [b for tr in other_tracks if _track_moved(tr, long_side * ROI_MOVE_RATIO) for _, b, _ in tr.hits]
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


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise SystemExit("ffmpeg not found — `pip install imageio-ffmpeg` or put ffmpeg on PATH")


def trim_video(src: Path, dst: Path, start_sec: float, end_sec: float) -> None:
    """Write [start_sec, end_sec] of `src` to `dst`: video re-encoded (H.264) so
    the cut lands on the exact frame (Blink keyframes are 4s apart, so a
    stream copy would keep up to 4s of extra lead-in), audio stream copied.
    The bitrate cap keeps the result smaller than the source. `src` is only read."""
    if dst.resolve() == src.resolve():
        raise ValueError(f"refusing to overwrite the source video: {src}")
    src_info = _video_info(src)
    src_kbps = src.stat().st_size * 8 / 1000 / max(0.1, src_info["duration_sec"])
    max_kbps = max(200, int(src_kbps * TRIM_MAX_RATE_RATIO))
    cmd = [
        _ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.3f}", "-i", str(src), "-t", f"{end_sec - start_sec:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", TRIM_PRESET, "-crf", str(TRIM_CRF),
        "-maxrate", f"{max_kbps}k", "-bufsize", f"{2 * max_kbps}k", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed ({proc.returncode}): {proc.stderr[-2000:]}")


def _video_info(path: Path) -> dict:
    """Same shape as describe_video.detect_segments()'s info dict."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    info = {
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "duration_sec": round(frame_count / fps, 3) if fps else 0.0,
    }
    cap.release()
    return info


def _write_jpeg(path: Path, frame: np.ndarray, max_side: int) -> tuple[int, int]:
    data = _jpeg_bytes(frame, max_side=max_side)
    path.write_bytes(data)
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    return int(w * scale), int(h * scale)


def sample_frames(
    video_path: Path,
    out_dir: Path,
    max_frames: int = MAX_FRAMES,
    roi: bool = True,
    info: Optional[dict] = None,
    segments: Optional[list] = None,
    trim: bool = True,
) -> Path:
    """Pass `info` + `segments` from a prior detect_segments() run (as
    describe_video.analyze() has) to skip a second OpenCV pass. With `trim`,
    writes <video stem>.trimmed.mp4 into out_dir and samples that instead."""
    if info is None or segments is None:
        info, segments, _ = detect_segments(video_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    source_info = info
    src_fps = info.get("fps") or 25.0
    src_duration = float(info.get("duration_sec") or 0.0)

    # Focus = while people are on screen, so a car passing before anyone
    # arrives (or after they leave) is neither kept in the trim nor sampled.
    # Clips with no person at all (e.g. a car arriving) fall back to motion.
    src_samples = scan_objects(video_path, src_fps, 0.0, src_duration) if segments else []
    src_people = build_tracks(src_samples, {"person"})
    if src_people:
        focus = (min(tr.hits[0][0] for tr in src_people), max(tr.hits[-1][0] for tr in src_people))
        focus_source = "person"
    elif segments:
        focus = (min(s.start_sec for s in segments), max(s.end_sec for s in segments))
        focus_source = "motion"
    else:
        focus, focus_source = None, None

    work_path = video_path
    offset = 0.0
    trim_meta = None
    if focus and trim:
        t0 = max(0.0, focus[0] - TRIM_PAD_SEC)
        t1 = min(src_duration, focus[1] + TRIM_PAD_SEC)
        work_path = out_dir / f"{video_path.stem}.trimmed.mp4"
        trim_video(video_path, work_path, t0, t1)
        info = _video_info(work_path)
        offset = t0
        trim_meta = {
            "file": work_path.name,
            "focus": focus_source,
            "offset_sec": round(t0, 3),
            "source_end_sec": round(t1, 3),
            "pad_sec": TRIM_PAD_SEC,
            "removed_head_sec": round(t0, 3),
            "removed_tail_sec": round(src_duration - t1, 3),
        }
        log.info("trimmed to %s presence: %.2fs -> %.2fs (cut %.2fs head, %.2fs tail) -> %s",
                 focus_source, src_duration, info["duration_sec"], t0, src_duration - t1, work_path)

    # Everything below is on the work video's timeline (trimmed when trimming).
    fps = info.get("fps") or 25.0
    duration = float(info.get("duration_sec") or 0.0)
    width, height = info.get("width") or 0, info.get("height") or 0
    last_frame = max(0, int(info.get("frame_count") or round(duration * fps)) - 1)
    # Shift + clip segments (copies — callers' objects untouched); drop motion
    # entirely outside the kept range. Indices stay the originals.
    segments = [
        replace(
            s,
            start_sec=round(max(0.0, s.start_sec - offset), 3),
            end_sec=round(min(duration, s.end_sec - offset), 3),
            keyframe_times_sec=[round(t - offset, 3) for t in s.keyframe_times_sec if 0 <= t - offset <= duration],
        )
        for s in segments
        if s.end_sec - offset > 0 and s.start_sec - offset < duration
    ]
    samples = [(round(t - offset, 3), dets) for t, dets in src_samples if 0 <= t - offset <= duration]
    log.info("video: %s - %d motion segment(s) in range", info, len(segments))

    manifest: dict = {
        "video": _video_ref(work_path),
        "info": info,
        "source_video": _video_ref(video_path),
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

    if focus:
        f0, f1 = max(0.0, focus[0] - offset), min(duration, focus[1] - offset)
        manifest["focus"] = {"source": focus_source, "start_sec": round(f0, 3), "end_sec": round(f1, 3)}
        span = (max(0.0, f0 - SPAN_PAD_SEC), min(duration, f1 + SPAN_PAD_SEC))
        manifest["span"] = {"start_sec": round(span[0], 3), "end_sec": round(span[1], 3)}

        tracks = build_tracks(samples, {"person"})
        other_tracks = {
            name: build_tracks(samples, {name}) for name in SCAN_CLASSES if name != "person"
        }
        triggers, crop_points = find_triggers(tracks, width, height)
        log.info("person scan: %d samples, %d track(s), %d trigger(s), %d bend(s)",
                 len(samples), len(tracks), len(triggers), len(crop_points))
        manifest["person_tracks"] = [
            {"id": tr.id, "first_sec": tr.hits[0][0], "last_sec": tr.hits[-1][0], "samples": len(tr.hits)}
            for tr in tracks
        ]
        long_side = max(width, height)
        manifest["object_tracks"] = [
            {"class": name, "id": tr.id, "first_sec": tr.hits[0][0], "last_sec": tr.hits[-1][0],
             "samples": len(tr.hits), "moved": _track_moved(tr, long_side * ROI_MOVE_RATIO)}
            for name, trs in other_tracks.items() for tr in trs
        ]

        # With people present, the crop follows them only — vehicles/animals
        # widen it only in person-less clips.
        roi_others = [] if tracks else [tr for trs in other_tracks.values() for tr in trs]
        roi_box, roi_source = (
            compute_roi(tracks, roi_others, segments, width, height) if roi else (None, "disabled")
        )
        if roi_box:
            area = (roi_box[2] - roi_box[0]) * (roi_box[3] - roi_box[1]) / float(width * height)
            manifest["roi"] = {"box": roi_box, "source": roi_source, "area_ratio": round(area, 3)}
            log.info("roi: %s from %s (%.0f%% of frame)", roi_box, roi_source, area * 100)
        else:
            log.info("roi: none (%s) - writing full frames", roi_source)
        # Same scale an uncropped frame would get, so cropping cuts pixels, not detail.
        frame_scale = min(1.0, FRAME_MAX_SIDE / max(1, long_side))
        manifest["triggers"] = [
            {"t": g.t, "reason": g.reason, "track": g.track, "score": g.score}
            for g in sorted(triggers + crop_points, key=lambda g: g.t)
        ]

        plan = plan_frames(span, triggers, fps, last_frame, max_frames)
        wanted = [p["frame_idx"] / fps for p in plan]
        decoded = _read_frames_at(work_path, wanted)

        for p in plan:
            frame = decoded.get(p["frame_idx"] / fps)
            if frame is None:
                log.warning("could not decode frame at %.2fs", p["t"])
                continue
            if roi_box:
                frame = frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]
            name = f"{p['t']:07.2f}s_{p['kind']}.jpg"
            w, h = _write_jpeg(out_dir / name, frame, int(round(max(frame.shape[:2]) * frame_scale)))
            manifest["frames"].append({
                "file": name, "t": p["t"], "kind": p["kind"], "reasons": p["reasons"],
                "segment": _segment_label(p["t"], segments), "width": w, "height": h,
                "approx_image_tokens": int(w * h * TOKENS_PER_PIXEL),
            })

        # Person close-up crops disabled - every frame gets the activity-region crop instead.
        # for g in bend_crops:
        #     t = round(g.t * fps) / fps
        #     frame = decoded.get(t)
        #     if frame is None:
        #         continue
        #     crop = _crop(frame, g.box)
        #     if crop.size == 0:
        #         continue
        #     name = f"{g.t:07.2f}s_crop_p{g.track}.jpg"
        #     w, h = _write_jpeg(out_dir / name, crop, CROP_MAX_SIDE)
        #     manifest["frames"].append({
        #         "file": name, "t": round(g.t, 3), "kind": "crop", "reasons": [f"bend-peak:p{g.track}"],
        #         "segment": _segment_label(g.t, segments), "width": w, "height": h,
        #         "approx_image_tokens": int(w * h * TOKENS_PER_PIXEL),
        #     })
    else:
        log.info("no motion segments — nothing to sample")

    frames = manifest["frames"]
    manifest["totals"] = {
        "frames": len(frames),
        "approx_image_tokens": sum(f["approx_image_tokens"] for f in frames),
    }
    out = out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %d frame(s) (~%d image tokens) -> %s",
             manifest["totals"]["frames"], manifest["totals"]["approx_image_tokens"], out_dir)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("-o", "--out-dir", type=Path, default=None,
                        help="Output directory (default: <video dir>/<video stem>.details/)")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="Cap on frames per clip")
    # Person close-up crops disabled - every frame gets the activity-region crop instead.
    # parser.add_argument("--no-crops", action="store_true", help="Skip high-res person crops at bend peaks")
    parser.add_argument("--no-roi", action="store_true", help="Write uncropped full frames")
    parser.add_argument("--no-trim", action="store_true", help="Sample the original video instead of a trimmed copy")
    args = parser.parse_args(argv)

    if not args.video.exists():
        log.error("video not found: %s", args.video)
        return 1
    out_dir = args.out_dir or args.video.parent / f"{args.video.stem}.details"
    sample_frames(args.video, out_dir, max_frames=args.max_frames, roi=not args.no_roi, trim=not args.no_trim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
