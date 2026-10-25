"""Motion segments: OpenCV background subtraction (MOG2) over the clip, grouped
into time spans of real movement, each with a union bounding box, a direction
and evenly spaced keyframes (for the YOLO check)."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from blink_watcher.settings import get_settings
from blink_watcher.vision.video import capture_fps, capture_info, open_video

_settings = get_settings()

log = logging.getLogger(__name__)

WARMUP_FRAMES = _settings.motion_warmup_frames                # feed MOG2 this many frames before trusting its output (absorbs the camera's auto-exposure flash)
LEARNING_RATE = _settings.motion_learning_rate                # low -> subjects that linger don't get absorbed as background
MIN_CONTOUR_AREA_RATIO = _settings.motion_min_contour_area_ratio  # a single contour must cover >=0.15% of frame to count (filters rain, leaves)
MIN_FRAME_AREA_RATIO = _settings.motion_min_frame_area_ratio  # sum of qualifying contours must cover >=0.25% of frame
MAX_FRAME_AREA_RATIO = _settings.motion_max_frame_area_ratio  # last-resort cap: close subjects can cover 30-60% of the frame; only near-total coverage is rejected
FLASH_MEAN_DELTA = _settings.motion_flash_mean_delta          # a grayscale mean change this large between sampled frames is an exposure / lighting flash, not motion
MIN_SEGMENT_SEC = _settings.motion_min_segment_sec            # drop blips shorter than this
MERGE_GAP_SEC = _settings.motion_merge_gap_sec                # stitch segments separated by smaller gaps
KEYFRAMES_PER_SEGMENT = _settings.motion_keyframes_per_segment  # evenly sampled moving frames
MOTION_MAX_SIDE = _settings.motion_max_side                   # downscale frames to this before MOG2 - big speedup, small subjects survive
MOTION_FRAME_STRIDE = _settings.motion_frame_stride           # analyze every Nth frame (30 fps source -> ~10 fps motion sampling)


@dataclass
class MotionSegment:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    bbox: tuple[int, int, int, int]  # x, y, w, h (union over the segment)
    peak_area_ratio: float
    keyframe_times_sec: list[float]
    direction: str = "unknown"  # "left to right", "right to left and downward", "stationary", ...
    centroid_start: Optional[tuple[int, int]] = None  # in original frame coords
    centroid_end: Optional[tuple[int, int]] = None
    detections: list[dict] = field(default_factory=list)  # moving YOLO objects (yolo.detect_moving_objects)

    def to_record(self) -> dict:
        """As stored in "motion_segments" of the metadata files."""
        return {**asdict(self), "bbox": list(self.bbox)}

    @classmethod
    def from_record(cls, seg: dict) -> "MotionSegment":
        bbox = seg.get("bbox") or [0, 0, 0, 0]
        cs, ce = seg.get("centroid_start"), seg.get("centroid_end")
        return cls(
            index=seg["index"],
            start_sec=seg["start_sec"],
            end_sec=seg["end_sec"],
            duration_sec=seg["duration_sec"],
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            peak_area_ratio=seg["peak_area_ratio"],
            keyframe_times_sec=list(seg.get("keyframe_times_sec") or []),
            direction=seg.get("direction", "unknown"),
            centroid_start=tuple(cs) if cs else None,
            centroid_end=tuple(ce) if ce else None,
            detections=list(seg.get("detections") or []),
        )


def keyframe_indices(segments: list[MotionSegment], fps: float) -> list[int]:
    return sorted({int(round(t * fps)) for s in segments for t in s.keyframe_times_sec})


def _classify_direction(cxs: list[float], cys: list[float], frame_w: int, frame_h: int) -> str:
    """The dominant direction of a subject from its per-frame centroids."""
    if len(cxs) < 2:
        return "unknown"
    dx = cxs[-1] - cxs[0]
    dy = cys[-1] - cys[0]
    # At least 5% of the frame dimension before claiming a direction.
    min_dx = frame_w * 0.05
    min_dy = frame_h * 0.05
    horiz = "left to right" if dx > min_dx else ("right to left" if dx < -min_dx else "")
    vert = "downward" if dy > min_dy else ("upward" if dy < -min_dy else "")
    if horiz and vert:
        return f"{horiz} and {vert}"
    return horiz or vert or "stationary"


def _frame_motion(mask: np.ndarray) -> tuple[float, Optional[tuple[int, int, int, int]], Optional[tuple[float, float]]]:
    """(area ratio, bbox, area-weighted centroid) of the meaningful moving
    blobs in a foreground mask."""
    h, w = mask.shape
    total = h * w
    # Erode away salt-pepper noise, then dilate generously to glue limbs / car
    # parts into a single blob that can be size-checked.
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
            continue  # rain drop, leaf, noise
        total_area += area
        cx, cy, cw, ch = cv2.boundingRect(c)
        x0, y0 = min(x0, cx), min(y0, cy)
        x1, y1 = max(x1, cx + cw), max(y1, cy + ch)
        cxsum += (cx + cw / 2) * area
        cysum += (cy + ch / 2) * area

    if total_area == 0:
        return 0.0, None, None
    return total_area / total, (x0, y0, x1 - x0, y1 - y0), (cxsum / total_area, cysum / total_area)


def detect_segments(video_path: Path) -> tuple[dict, list[MotionSegment]]:
    """Scan the clip once: (video info, motion segments)."""
    cap = open_video(video_path)
    info = capture_info(cap)
    fps, width, height = capture_fps(cap), info["width"], info["height"]
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

    # Every MOTION_FRAME_STRIDE-th frame is decoded and analyzed; the others
    # are skipped with grab() (no decode). per_sample[i] is the motion of
    # analyzed frame i, sample_to_frame[i] its frame number.
    per_sample: list[tuple[float, Optional[tuple[int, int, int, int]], Optional[tuple[float, float]]]] = []
    sample_to_frame: list[int] = []

    # MOG2 on 1920x1080 is ~10x slower than on 640x360. Contour ratios are
    # frame-relative, so only bboxes need scaling back to original coords.
    motion_scale = min(1.0, MOTION_MAX_SIDE / max(width, height)) if max(width, height) else 1.0
    inv_scale = 1.0 / motion_scale if motion_scale > 0 else 1.0
    stride = max(1, MOTION_FRAME_STRIDE)
    warmup_samples = max(1, WARMUP_FRAMES // stride)  # about the same wall-clock warmup despite striding

    idx = 0
    prev_mean: Optional[float] = None
    try:
        while True:
            if idx % stride:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            small = (cv2.resize(frame, (0, 0), fx=motion_scale, fy=motion_scale, interpolation=cv2.INTER_AREA)
                     if motion_scale < 1.0 else frame)
            # A global brightness jump between sampled frames is an exposure
            # flash: the whole scene shifts without anything moving.
            gray_mean = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
            is_flash = prev_mean is not None and abs(gray_mean - prev_mean) >= FLASH_MEAN_DELTA
            prev_mean = gray_mean
            mask = subtractor.apply(small, learningRate=LEARNING_RATE)
            if len(per_sample) < warmup_samples or is_flash:
                per_sample.append((0.0, None, None))
            else:
                per_sample.append(_frame_motion(mask))
            sample_to_frame.append(idx)
            idx += 1
    finally:
        cap.release()

    # Consecutive moving samples -> raw segments (indices into per_sample).
    raw: list[list[int]] = []
    current: list[int] = []
    for si, (area, _, _) in enumerate(per_sample):
        if MIN_FRAME_AREA_RATIO <= area <= MAX_FRAME_AREA_RATIO:
            current.append(si)
        elif current:
            raw.append(current)
            current = []
    if current:
        raw.append(current)

    # Merge segments separated by small gaps. Only the moving samples are
    # carried across: gap samples can be exposure flashes above
    # MAX_FRAME_AREA_RATIO and would pollute the union bbox / peak ratio.
    merge_gap_samples = max(1, int(MERGE_GAP_SEC * fps / stride))
    merged: list[list[int]] = []
    for seg in raw:
        if merged and seg[0] - merged[-1][-1] <= merge_gap_samples:
            merged[-1].extend(seg)
        else:
            merged.append(list(seg))

    min_len_samples = max(1, int(MIN_SEGMENT_SEC * fps / stride))
    merged = [s for s in merged if len(s) >= min_len_samples]

    small_w = int(width * motion_scale) if motion_scale < 1.0 else width
    small_h = int(height * motion_scale) if motion_scale < 1.0 else height
    segments: list[MotionSegment] = []
    for i, seg in enumerate(merged):
        moving = [si for si in seg if per_sample[si][1] is not None]
        if not moving:
            continue
        boxes = [per_sample[si][1] for si in moving]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
        peak = max(per_sample[si][0] for si in moving)

        # Centroid trajectory (downscaled coords) -> direction.
        centroids = [per_sample[si][2] for si in moving if per_sample[si][2] is not None]
        cxs, cys = [c[0] for c in centroids], [c[1] for c in centroids]
        direction = _classify_direction(cxs, cys, small_w, small_h)
        if cxs:
            c_start = (int(cxs[0] * inv_scale), int(cys[0] * inv_scale))
            c_end = (int(cxs[-1] * inv_scale), int(cys[-1] * inv_scale))
        else:
            c_start = c_end = None

        # Keyframes evenly spread over the moving samples.
        if len(moving) >= KEYFRAMES_PER_SEGMENT:
            step = (len(moving) - 1) / (KEYFRAMES_PER_SEGMENT - 1)
            kf_samples = [moving[int(round(k * step))] for k in range(KEYFRAMES_PER_SEGMENT)]
        else:
            kf_samples = moving
        kf_frames = [sample_to_frame[si] for si in kf_samples]

        first_frame, last_frame = sample_to_frame[seg[0]], sample_to_frame[seg[-1]]
        segments.append(MotionSegment(
            index=i,
            start_sec=round(first_frame / fps, 3),
            end_sec=round(last_frame / fps, 3),
            duration_sec=round((last_frame - first_frame + 1) / fps, 3),
            bbox=(int(x0 * inv_scale), int(y0 * inv_scale), int((x1 - x0) * inv_scale), int((y1 - y0) * inv_scale)),
            peak_area_ratio=round(peak, 4),
            keyframe_times_sec=[round(k / fps, 3) for k in kf_frames],
            direction=direction,
            centroid_start=c_start,
            centroid_end=c_end,
        ))
    return info, segments
