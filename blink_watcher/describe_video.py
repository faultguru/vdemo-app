"""Analyze a video and emit a metadata JSON describing what is moving.

Pipeline: OpenCV background subtraction → motion segments → keyframe sampling
→ Claude Opus 4.6 vision call → structured JSON written next to the video.

Claude is authenticated via the OAuth access token stored in
``~/.claude/.credentials.json`` (the Claude Code credentials file).

Usage:
    python -m blink_watcher.describe_video path/to/clip.mp4
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv

from blink_watcher.describers import (
    VideoAnalysis,
    _empty_summary,
    _jpeg_bytes,
    make_describer,
)

log = logging.getLogger("blink_watcher.describe_video")

load_dotenv()

# Motion detection knobs.
WARMUP_FRAMES = 35             # feed MOG2 this many frames before trusting its output (absorbs Blink auto-exposure flash)
LEARNING_RATE = 0.0008         # low → subjects that linger don't get absorbed as background
MIN_CONTOUR_AREA_RATIO = 0.0015# a single contour must cover ≥0.15% of frame to count (filters rain, leaves)
MIN_FRAME_AREA_RATIO = 0.0025  # sum of qualifying contours must cover ≥0.25% of frame
MAX_FRAME_AREA_RATIO = 0.90    # last-resort cap. Real close-range subjects can cover 30–60% of the frame; only reject near-total coverage.
FLASH_MEAN_DELTA = 12.0        # if grayscale mean intensity changes by this much between consecutive sampled frames, treat it as a global exposure/lighting flash and drop that sample
MIN_SEGMENT_SEC = 0.4          # drop blips shorter than this
MERGE_GAP_SEC = 1.0            # stitch segments separated by smaller gaps
MAX_SEGMENTS_TO_DESCRIBE = 12  # cap Claude call cost
KEYFRAMES_PER_SEGMENT = 8      # evenly sampled moving frames
KEYFRAME_MAX_SIDE = 1600       # keyframe JPEG resolution sent to Claude
MOTION_MAX_SIDE = 960          # downscale frames to this max side before MOG2 — big speedup while preserving small subjects
MOTION_FRAME_STRIDE = 3        # only decode + analyze every Nth frame in pass 1 (30fps source → ~10fps motion sampling)
TRIAGE_SAMPLES = 6             # number of evenly-spaced frames decoded for the fast pre-pass
TRIAGE_MAX_SIDE = 160          # downscale triage frames to this max side — absdiff at tiny resolution is near-free
TRIAGE_PIXEL_DELTA = 20        # per-pixel intensity delta that counts as "changed"
TRIAGE_CHANGED_RATIO = 0.005   # if at least this fraction of pixels changed between any two samples, treat as real motion (0.5%)
YOLO_MODEL_NAME = "yolov8s.pt" # small variant — ~22MB, more accurate than nano
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
YOLO_CONF_THRESHOLD = 0.20     # min confidence to report a detection
# COCO classes that can plausibly move in a residential surveillance scene.
# Everything else (tv, chair, potted plant, bench, ...) is static scenery — drop
# it even if YOLO lands a high-confidence detection on top of the motion bbox.
YOLO_MOVING_CLASSES = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "bird", "cat", "dog", "horse", "sheep", "cow",
    "skateboard", "sports ball", "frisbee",
})


# ---------- Fast triage pass ----------


def triage_video(video_path: Path) -> tuple[dict, bool, float]:
    """Cheap pre-pass: seek-decode TRIAGE_SAMPLES frames spread across the clip,
    downscale to TRIAGE_MAX_SIDE, and return whether any real change happened.

    Uses global grayscale-mean delta to reject Blink exposure flashes (same
    rule as the main motion pass) so brightness flashes don't trip triage.

    Returns (info, passed, max_delta). `info` has fps/width/height/duration/
    frame_count so the caller doesn't need to re-open the file for metadata.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    info = {
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(frame_count / fps, 3) if frame_count else None,
    }

    if frame_count <= 0:
        cap.release()
        return info, True, 0.0  # unknown → let the full pipeline handle it

    # Pick TRIAGE_SAMPLES evenly spaced frame indices, avoiding the very first
    # frame (which can be a Blink exposure flash before auto-exposure settles).
    start = max(1, frame_count // 20)
    step = max(1, (frame_count - start - 1) // max(1, TRIAGE_SAMPLES - 1))
    targets = [min(frame_count - 1, start + i * step) for i in range(TRIAGE_SAMPLES)]

    scale = min(1.0, TRIAGE_MAX_SIDE / max(width, height)) if max(width, height) else 1.0
    grays: list[np.ndarray] = []
    means: list[float] = []
    for t in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        if not ok:
            continue
        if scale < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(g)
        means.append(float(g.mean()))
    cap.release()

    if len(grays) < 2:
        return info, True, 0.0  # couldn't sample enough — don't skip

    # Compare every pair of samples — a subject visible in just one mid-clip
    # frame still pops against start + end baselines. For each pair, count the
    # fraction of pixels whose intensity shifted by more than TRIAGE_PIXEL_DELTA.
    # Skip pairs whose global mean intensity moved (that's a flash, not motion).
    max_ratio = 0.0
    n = len(grays)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(means[i] - means[j]) >= FLASH_MEAN_DELTA:
                continue
            diff = cv2.absdiff(grays[i], grays[j])
            changed = float((diff >= TRIAGE_PIXEL_DELTA).mean())
            if changed > max_ratio:
                max_ratio = changed

    return info, max_ratio >= TRIAGE_CHANGED_RATIO, max_ratio


# ---------- Motion detection ----------

@dataclass
class MotionSegment:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    bbox: tuple[int, int, int, int]  # x, y, w, h (union over the segment)
    peak_area_ratio: float
    keyframe_times_sec: list[float]
    direction: str = "unknown"  # human-readable: "left to right", "right to left", "up", "down", "stationary"
    centroid_start: Optional[tuple[int, int]] = None  # in original frame coords
    centroid_end: Optional[tuple[int, int]] = None
    detections: list[dict] = None  # YOLO detections, populated by detect_objects()

    def __post_init__(self):
        if self.detections is None:
            self.detections = []


def _classify_direction(
    cxs: list[float], cys: list[float], frame_w: int, frame_h: int
) -> str:
    """Classify the dominant direction of a subject from a list of per-frame centroids."""
    if len(cxs) < 2:
        return "unknown"
    dx = cxs[-1] - cxs[0]
    dy = cys[-1] - cys[0]
    # Require at least 5% of frame dimension of movement before claiming direction.
    min_dx = frame_w * 0.05
    min_dy = frame_h * 0.05
    horiz = "left to right" if dx > min_dx else ("right to left" if dx < -min_dx else "")
    vert = "downward" if dy > min_dy else ("upward" if dy < -min_dy else "")
    if horiz and vert:
        return f"{horiz} and {vert}"
    return horiz or vert or "stationary"


def _frame_motion(mask: np.ndarray) -> tuple[float, Optional[tuple[int, int, int, int]], Optional[tuple[float, float]]]:
    """Return (area_ratio, bbox, centroid) for meaningful moving objects in the
    foreground mask. Centroid is the area-weighted centroid of qualifying
    contours — used to track direction across a segment."""
    h, w = mask.shape
    total = h * w
    # Erode away salt-pepper noise, then dilate generously to glue limbs / car
    # parts into a single blob we can size-check.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_pixels = MIN_CONTOUR_AREA_RATIO * total

    total_area = 0.0
    x0 = y0 = 10**9
    x1 = y1 = -1
    cxsum = cysum = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_contour_pixels:
            continue  # too small → rain drop, leaf, noise
        total_area += area
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cx < x0: x0 = cx
        if cy < y0: y0 = cy
        if cx + cw > x1: x1 = cx + cw
        if cy + ch > y1: y1 = cy + ch
        cxsum += (cx + cw / 2) * area
        cysum += (cy + ch / 2) * area

    if total_area == 0:
        return 0.0, None, None
    centroid = (cxsum / total_area, cysum / total_area)
    return total_area / total, (x0, y0, x1 - x0, y1 - y0), centroid


def detect_segments(video_path: Path) -> tuple[dict, list[MotionSegment], dict[int, np.ndarray]]:
    """Scan the video once. Return video info, motion segments, and a cache of
    decoded BGR frames for the keyframe indices we'll send to Claude."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 25.0  # fallback for broken headers
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    info = {
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(frame_count / fps, 3) if frame_count else None,
    }

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=40, detectShadows=False
    )

    # Pass 1 processes every Nth frame (MOTION_FRAME_STRIDE). Skipped frames
    # are advanced via cap.grab() — no decode cost, no MOG2 cost. Each entry
    # in per_sample corresponds to one analyzed frame; sample_to_frame maps a
    # sample index back to the original frame number.
    per_sample: list[tuple[float, Optional[tuple[int, int, int, int]], Optional[tuple[float, float]]]] = []
    sample_to_frame: list[int] = []
    frames_bgr: dict[int, np.ndarray] = {}  # populated in pass 2

    # Downscale factor for the motion pass — MOG2 on 1920x1080 is ~10x slower
    # than on 640x360. Contour ratios are frame-relative so filtering still
    # works; we only need to scale bboxes back to original coords.
    motion_scale = min(1.0, MOTION_MAX_SIDE / max(width, height)) if max(width, height) else 1.0
    inv_scale = 1.0 / motion_scale if motion_scale > 0 else 1.0

    stride = max(1, MOTION_FRAME_STRIDE)
    # Keep warmup roughly the same wall-clock duration despite striding.
    warmup_samples = max(1, WARMUP_FRAMES // stride)

    idx = 0
    prev_mean: Optional[float] = None
    while True:
        if idx % stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            if motion_scale < 1.0:
                small = cv2.resize(frame, (0, 0), fx=motion_scale, fy=motion_scale, interpolation=cv2.INTER_AREA)
            else:
                small = frame
            # Global brightness delta between consecutive sampled frames — detects
            # Blink's exposure/aperture flashes where the whole scene's intensity
            # shifts without any real subject moving.
            gray_mean = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
            is_flash = prev_mean is not None and abs(gray_mean - prev_mean) >= FLASH_MEAN_DELTA
            prev_mean = gray_mean
            mask = subtractor.apply(small, learningRate=LEARNING_RATE)
            if len(per_sample) < warmup_samples or is_flash:
                per_sample.append((0.0, None, None))
            else:
                per_sample.append(_frame_motion(mask))
            sample_to_frame.append(idx)
        else:
            # Advance the reader without decoding this frame.
            if not cap.grab():
                break
        idx += 1
    cap.release()

    # Group consecutive moving SAMPLES into raw segments (indices into per_sample).
    raw: list[list[int]] = []
    current: list[int] = []
    for si, tpl in enumerate(per_sample):
        area = tpl[0]
        if MIN_FRAME_AREA_RATIO <= area <= MAX_FRAME_AREA_RATIO:
            current.append(si)
        elif current:
            raw.append(current)
            current = []
    if current:
        raw.append(current)

    # Merge segments separated by small gaps (in samples now). We only carry
    # the real moving samples across the merge — the gap samples are NOT
    # included, because they can be above MAX_FRAME_AREA_RATIO (exposure
    # flashes) and would pollute the union bbox / peak ratio with whole-frame
    # lighting changes.
    merge_gap_samples = max(1, int(MERGE_GAP_SEC * fps / stride))
    merged: list[list[int]] = []
    for seg in raw:
        if merged and seg[0] - merged[-1][-1] <= merge_gap_samples:
            merged[-1].extend(seg)
        else:
            merged.append(list(seg))

    # Drop tiny segments.
    min_len_samples = max(1, int(MIN_SEGMENT_SEC * fps / stride))
    merged = [s for s in merged if len(s) >= min_len_samples]

    segments: list[MotionSegment] = []
    for i, seg in enumerate(merged):
        moving_samples = [si for si in seg if per_sample[si][1] is not None]
        if not moving_samples:
            continue
        # Union bbox + peak area.
        x0 = min(per_sample[si][1][0] for si in moving_samples)
        y0 = min(per_sample[si][1][1] for si in moving_samples)
        x1 = max(per_sample[si][1][0] + per_sample[si][1][2] for si in moving_samples)
        y1 = max(per_sample[si][1][1] + per_sample[si][1][3] for si in moving_samples)
        peak = max(per_sample[si][0] for si in moving_samples)

        # Centroid trajectory (in downscaled coords) → classify direction.
        cxs = [per_sample[si][2][0] for si in moving_samples if per_sample[si][2] is not None]
        cys = [per_sample[si][2][1] for si in moving_samples if per_sample[si][2] is not None]
        small_w = int(width * motion_scale) if motion_scale < 1.0 else width
        small_h = int(height * motion_scale) if motion_scale < 1.0 else height
        direction = _classify_direction(cxs, cys, small_w, small_h)
        if cxs:
            c_start = (int(cxs[0] * inv_scale), int(cys[0] * inv_scale))
            c_end = (int(cxs[-1] * inv_scale), int(cys[-1] * inv_scale))
        else:
            c_start = c_end = None

        # Pick keyframes evenly from moving samples, then map back to frame idx.
        if len(moving_samples) >= KEYFRAMES_PER_SEGMENT:
            step = (len(moving_samples) - 1) / (KEYFRAMES_PER_SEGMENT - 1)
            kf_samples = [moving_samples[int(round(k * step))] for k in range(KEYFRAMES_PER_SEGMENT)]
        else:
            kf_samples = moving_samples
        kf_frames = [sample_to_frame[si] for si in kf_samples]
        for k in kf_frames:
            frames_bgr[k] = None  # placeholder, filled in pass 2

        first_frame = sample_to_frame[seg[0]]
        last_frame = sample_to_frame[seg[-1]]

        # Scale bbox from downscaled-motion coords back to original frame coords.
        bx = int(x0 * inv_scale)
        by = int(y0 * inv_scale)
        bw = int((x1 - x0) * inv_scale)
        bh = int((y1 - y0) * inv_scale)
        segments.append(MotionSegment(
            index=i,
            start_sec=round(first_frame / fps, 3),
            end_sec=round(last_frame / fps, 3),
            duration_sec=round((last_frame - first_frame + 1) / fps, 3),
            bbox=(bx, by, bw, bh),
            peak_area_ratio=round(peak, 4),
            keyframe_times_sec=[round(k / fps, 3) for k in kf_frames],
            direction=direction,
            centroid_start=c_start,
            centroid_end=c_end,
        ))

    # Pass 2: seek-decode only the keyframes we need.
    if frames_bgr:
        cap = cv2.VideoCapture(str(video_path))
        for target in sorted(frames_bgr.keys()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if ok:
                frames_bgr[target] = frame
        cap.release()

    return info, segments, frames_bgr


def load_prior_opencv(json_path: Path) -> Optional[tuple[dict, list[MotionSegment], dict]]:
    """Return (info, segments, processors_by_name) from a prior run's JSON if
    opencv already ran. Returns None if the file doesn't exist, is unreadable,
    or has no opencv processor."""
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    processors = {p.get("name"): p for p in data.get("processors", [])}
    if "opencv" not in processors:
        return None
    info = data.get("info") or {}
    segments: list[MotionSegment] = []
    for seg in data.get("motion_segments", []):
        bbox = seg.get("bbox") or [0, 0, 0, 0]
        cs = seg.get("centroid_start")
        ce = seg.get("centroid_end")
        segments.append(MotionSegment(
            index=seg["index"],
            start_sec=seg["start_sec"],
            end_sec=seg["end_sec"],
            duration_sec=seg["duration_sec"],
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            peak_area_ratio=seg["peak_area_ratio"],
            keyframe_times_sec=list(seg.get("keyframe_times_sec", [])),
            direction=seg.get("direction", "unknown"),
            centroid_start=tuple(cs) if cs else None,
            centroid_end=tuple(ce) if ce else None,
            detections=list(seg.get("detections", [])),
        ))
    return info, segments, processors


def extract_keyframes(video_path: Path, segments: list[MotionSegment], fps: float) -> dict[int, np.ndarray]:
    """Seek-only decode of the keyframes referenced by the given segments."""
    frames_bgr: dict[int, np.ndarray] = {}
    if not segments:
        return frames_bgr
    wanted = sorted({int(round(t * fps)) for s in segments for t in s.keyframe_times_sec})
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    for target in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if ok:
            frames_bgr[target] = frame
    cap.release()
    return frames_bgr


# ---------- YOLO object detection ----------

_yolo_model = None


def _get_yolo():
    """Load the YOLOv8 model once and cache it."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        model_path = MODELS_DIR / YOLO_MODEL_NAME
        log.info("loading YOLO model %s", model_path)
        _yolo_model = YOLO(str(model_path))
    return _yolo_model


def _bbox_iou_motion(det_xyxy: list[float], motion_xywh: tuple[int, int, int, int]) -> float:
    """Overlap ratio of a YOLO detection (x1,y1,x2,y2) against the motion region
    (x,y,w,h), measured as (intersection / detection area). > 0 means the
    detection is at least partly inside the moving region."""
    dx1, dy1, dx2, dy2 = det_xyxy
    mx, my, mw, mh = motion_xywh
    mx2 = mx + mw
    my2 = my + mh
    ix1 = max(dx1, mx)
    iy1 = max(dy1, my)
    ix2 = min(dx2, mx2)
    iy2 = min(dy2, my2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    det_area = max(1.0, (dx2 - dx1) * (dy2 - dy1))
    return inter / det_area


def detect_objects(
    segments: list[MotionSegment],
    frames_bgr: dict[int, np.ndarray],
    fps: float,
    info: dict,
) -> dict[int, list[dict]]:
    """Run YOLO on each segment's keyframes and return ONLY detections that
    actually move across the keyframes. A class passes if either:
      (a) its highest-confidence detection across keyframes shifts center
          by > 3% of the frame's max side (proves it moved), or
      (b) it is present in only one keyframe but that is a keyframe near the
          segment's first or last frame (entered/left the scene).
    """
    if not segments or not frames_bgr:
        return {}
    model = _get_yolo()
    names = model.names
    frame_max_side = max(info.get("width") or 0, info.get("height") or 0) or 1
    min_shift_px = frame_max_side * 0.03  # 3% of long side

    per_segment: dict[int, list[dict]] = {}
    for seg in segments:
        # Collect frames + the ordinal index of each keyframe in the segment.
        kf_list: list[tuple[int, np.ndarray]] = []  # (kf_ord, frame)
        for kf_ord, t_sec in enumerate(seg.keyframe_times_sec):
            frame_idx = int(round(t_sec * fps))
            frame = frames_bgr.get(frame_idx)
            if frame is not None:
                kf_list.append((kf_ord, frame))
        if not kf_list:
            per_segment[seg.index] = []
            continue

        results = model.predict(
            [f for _, f in kf_list],
            conf=YOLO_CONF_THRESHOLD,
            verbose=False,
        )

        # Per class, collect a list of (kf_ord, conf, xyxy) across keyframes.
        per_class: dict[str, list[tuple[int, float, list[float]]]] = {}
        for (kf_ord, _frame), r in zip(kf_list, results):
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                name = names.get(cls_id, str(cls_id))
                if name not in YOLO_MOVING_CLASSES:
                    continue
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                iou = _bbox_iou_motion(xyxy, seg.bbox)
                if iou < 0.2:
                    continue
                per_class.setdefault(name, []).append((kf_ord, conf, xyxy))

        # Keep only classes that genuinely moved across keyframes.
        kept: list[dict] = []
        last_kf = len(seg.keyframe_times_sec) - 1
        for name, hits in per_class.items():
            # For each keyframe, take the highest-conf detection for that class.
            best_per_kf: dict[int, tuple[float, list[float]]] = {}
            for kf_ord, conf, xyxy in hits:
                prev = best_per_kf.get(kf_ord)
                if prev is None or conf > prev[0]:
                    best_per_kf[kf_ord] = (conf, xyxy)

            centers = {
                kf_ord: ((v[1][0] + v[1][2]) / 2, (v[1][1] + v[1][3]) / 2)
                for kf_ord, v in best_per_kf.items()
            }
            if len(centers) >= 2:
                xs = [c[0] for c in centers.values()]
                ys = [c[1] for c in centers.values()]
                shift = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
                moved = shift >= min_shift_px
            else:
                # Single-keyframe detection: only accept if it's at an edge of
                # the segment (entered or left the scene during the window).
                only_kf = next(iter(centers))
                moved = only_kf == 0 or only_kf == last_kf
                shift = 0.0

            if not moved:
                continue
            # Pick the best-confidence detection overall for this class.
            best_kf_ord, (best_conf, best_xyxy) = max(
                best_per_kf.items(), key=lambda kv: kv[1][0]
            )
            kept.append({
                "class": name,
                "confidence": round(best_conf, 3),
                "bbox": [int(v) for v in best_xyxy],
                "keyframe_shift_px": int(shift),
                "present_in_keyframes": sorted(best_per_kf.keys()),
            })

        per_segment[seg.index] = sorted(kept, key=lambda d: -d["confidence"])
    return per_segment


# ---------- Entry point ----------

def _video_ref(video_path: Path) -> str:
    """Return the clip's path relative to the nearest 'videos' ancestor, using
    forward slashes. Falls back to the bare filename if no such ancestor."""
    resolved = video_path.resolve()
    parts = resolved.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "videos":
            return "/".join(parts[i + 1:])
    return resolved.name


def _read_frames_at(video_path: Path, timestamps_sec: list[float]) -> dict[float, np.ndarray]:
    """Seek the raw video and return a dict {timestamp: BGR frame} for each
    requested time, in the order they appear. Missing/unreadable seeks are
    skipped."""
    if not timestamps_sec:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out: dict[float, np.ndarray] = {}
    for t in sorted(set(timestamps_sec)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if ok:
            out[t] = frame
    cap.release()
    return out


def analyze(
    video_path: Path,
    output_path: Optional[Path] = None,
    motion_only: bool = False,
) -> Path:
    out = output_path or video_path.with_suffix(".json")

    prior = load_prior_opencv(out)
    processors: list[dict] = []
    prior_procs: dict = {}
    triage_skipped = False
    if prior is not None:
        info, segments, prior_procs = prior
        log.info("reusing opencv result from %s (%d segment(s))", out.name, len(segments))
        fps = info.get("fps") or 25.0
        frames = extract_keyframes(video_path, segments, fps)
        processors.append({**prior_procs["opencv"], "reused": True})
    else:
        # Fast triage pass — if nothing visibly changes between sparse samples,
        # skip the full MOG2 scan + YOLO + LLM call entirely.
        t0 = time.perf_counter()
        info, triage_passed, triage_delta = triage_video(video_path)
        triage_ms = int((time.perf_counter() - t0) * 1000)
        log.info("triage: ratio=%.4f passed=%s in %d ms", triage_delta, triage_passed, triage_ms)
        processors.append({
            "name": "triage",
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_ms": triage_ms,
            "changed_ratio": round(triage_delta, 4),
            "threshold": TRIAGE_CHANGED_RATIO,
            "passed": triage_passed,
        })
        if not triage_passed:
            triage_skipped = True
            segments = []
            frames = {}
            log.info("video: %s", info)
            log.info("triage short-circuit — no further processing")
        else:
            t0 = time.perf_counter()
            info, segments, frames = detect_segments(video_path)
            opencv_ms = int((time.perf_counter() - t0) * 1000)
            log.info("video: %s", info)
            log.info("found %d motion segment(s) in %d ms", len(segments), opencv_ms)
            processors.append({
                "name": "opencv",
                "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "duration_ms": opencv_ms,
            })

    # YOLO pass — reused from prior run if present, otherwise run fresh.
    if segments and not motion_only:
        if "yolo" in prior_procs:
            log.info("reusing yolo result")
            processors.append({**prior_procs["yolo"], "reused": True})
            # seg.detections were already restored in load_prior_opencv
        else:
            fps = info.get("fps") or 25.0
            t0 = time.perf_counter()
            dets_by_seg = detect_objects(segments, frames, fps, info)
            yolo_ms = int((time.perf_counter() - t0) * 1000)
            classes_seen: set[str] = set()
            for seg in segments:
                seg.detections = dets_by_seg.get(seg.index, [])
                for d in seg.detections:
                    classes_seen.add(d["class"])
                # Direction from centroid tracking is only meaningful when
                # there's a real moving subject. Without YOLO confirmation,
                # the centroid is drifting across noise — clear it.
                if not seg.detections:
                    seg.direction = "unknown"
                    seg.centroid_start = None
                    seg.centroid_end = None
            classes_detected = sorted(classes_seen)
            log.info("yolo detected classes %s in %d ms", classes_detected, yolo_ms)
            processors.append({
                "name": "yolo",
                "model": YOLO_MODEL_NAME,
                "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "duration_ms": yolo_ms,
                "classes_detected": classes_detected,
            })

    any_detections = any(seg.detections for seg in segments)
    if motion_only:
        log.info("motion-only mode: skipping LLM description")
        analysis = VideoAnalysis(summary=_empty_summary("", ignore=True), events=[])
    elif segments and not any_detections:
        log.info("no YOLO detections on any segment — skipping LLM")
        analysis = VideoAnalysis(summary=_empty_summary("No motion.", ignore=True), events=[])
    elif segments:
        describer = make_describer()
        t0 = time.perf_counter()
        analysis, usage = describer.describe(video_path, segments, frames, info)
        llm_ms = int((time.perf_counter() - t0) * 1000)
        processors.append({
            "name": "llm",
            "provider": describer.provider,
            "model": describer.model,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_ms": llm_ms,
            "usage": usage,
        })
        # Auto-ignore clips the LLM resolved to nothing: no description, no
        # people, no vehicles, no actions. These are noise we don't want to
        # surface as events.
        s = analysis.summary
        desc = (s.description or "").strip().lower().rstrip(".")
        if (
            desc in ("", "no motion", "no motion detected")
            and not s.people
            and not s.vehicles
            and not s.actions
        ):
            log.info("llm returned no real content — marking clip as ignore")
            s.ignore = True
    else:
        analysis = VideoAnalysis(summary=_empty_summary("No motion detected.", ignore=True), events=[])

    # Dump important frames + thumbnail whenever the clip is NOT marked ignore.
    # ignore=True clips are no-motion / skipped-LLM — no thumbnail, no keyframes.
    kf_dir = video_path.parent / f"{video_path.stem}.keyframes"
    thumb_path = video_path.with_suffix(".jpg")

    if not analysis.summary.ignore:
        llm_frames = sorted(set(t for t in analysis.important_frames_sec if t >= 0))
        # Fallback: LLM ran but gave us no important_frames_sec — use each
        # segment's middle keyframe so we still produce a usable thumbnail.
        if not llm_frames and segments:
            llm_frames = sorted({
                seg.keyframe_times_sec[len(seg.keyframe_times_sec) // 2]
                for seg in segments
                if seg.keyframe_times_sec
            })

        extracted = _read_frames_at(video_path, llm_frames) if llm_frames else {}
        if extracted:
            kf_dir.mkdir(exist_ok=True)
            for t_sec, frame in extracted.items():
                (kf_dir / f"{t_sec:07.2f}s.jpg").write_bytes(_jpeg_bytes(frame))
            log.info("wrote %d keyframes to %s", len(extracted), kf_dir)

            thumb_t = analysis.thumbnail_sec
            thumb_frame = extracted.get(thumb_t)
            if thumb_frame is None and thumb_t >= 0:
                thumb_frame = next(iter(_read_frames_at(video_path, [thumb_t]).values()), None)
            if thumb_frame is None:
                # No llm-picked thumbnail — use the middle keyframe so we
                # still emit a thumbnail whenever ignore is False.
                mid = sorted(extracted.keys())[len(extracted) // 2]
                thumb_frame = extracted[mid]
                thumb_t = mid
            thumb_path.write_bytes(_jpeg_bytes(thumb_frame))
            log.info("wrote thumbnail %s (t=%.2fs)", thumb_path, thumb_t)
        else:
            log.warning("ignore=False but no frames available for thumbnail")

    seg_lookup = {s.index: s for s in segments}
    events_out = []
    for ev in analysis.events:
        seg = seg_lookup.get(ev.segment_index)
        events_out.append({
            "segment_index": ev.segment_index,
            "start_sec": seg.start_sec if seg else None,
            "end_sec": seg.end_sec if seg else None,
            "bbox": list(seg.bbox) if seg else None,
            "object": ev.object,
            "details": ev.details,
            "description": ev.description,
        })

    segments_out = []
    for seg in segments:
        entry = asdict(seg)
        entry["bbox"] = list(seg.bbox)
        segments_out.append(entry)

    metadata = {
        "video": _video_ref(video_path),
        "info": info,
        "summary": analysis.summary.model_dump(),
        "processors": processors,
        "thumbnail_sec": analysis.thumbnail_sec if analysis.thumbnail_sec >= 0 else None,
        "important_frames_sec": analysis.important_frames_sec,
        "events": events_out,
        "motion_segments": segments_out,
    }

    out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("wrote %s", out)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--motion-only", action="store_true", help="Only run OpenCV motion detection; skip the Claude description call")
    args = parser.parse_args(argv)

    if not args.video.exists():
        log.error("video not found: %s", args.video)
        return 1
    analyze(args.video, args.output, motion_only=args.motion_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
