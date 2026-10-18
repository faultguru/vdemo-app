"""The check pass: does anything worth describing happen in a clip? Cheapest
step first, stopping at the first that says no:

  triage  - do sampled frames change at all?            (triage.py)
  opencv  - motion segments                             (motion.py)
  yolo    - a moving object of the required classes     (yolo.py)
            in the segments' keyframes

Each step leaves a processor record ({"name", "completed_at", "duration_ms",
... "passed"}) as stored in "processors" of the metadata files.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from blink_watcher.vision.motion import MotionSegment, detect_segments, keyframe_indices
from blink_watcher.vision.triage import TRIAGE_CHANGED_RATIO, triage_video
from blink_watcher.vision.video import read_frames, video_ref
from blink_watcher.vision.yolo import YOLO_MODEL_NAME, detect_moving_objects, yolo_verdict

log = logging.getLogger(__name__)

# progress(processor record, motion segments so far), after each step
CheckProgress = Callable[[dict, list[MotionSegment]], None]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CheckResult:
    video: str                       # video.video_ref of the checked clip
    info: dict
    segments: list[MotionSegment] = field(default_factory=list)
    processors: list[dict] = field(default_factory=list)

    def record(self) -> dict:
        return {
            "video": self.video,
            "info": self.info,
            "processors": self.processors,
            "motion_segments": [s.to_record() for s in self.segments],
        }


def check_video(video_path: Path, required_classes: Optional[list[str]] = None,
                progress: Optional[CheckProgress] = None) -> CheckResult:
    """Triage -> OpenCV motion -> YOLO; `required_classes`: the YOLO step
    passes only if one of them moves (empty = any class)."""
    def done(proc: dict, segments: list[MotionSegment]) -> None:
        result.processors.append(proc)
        if progress:
            progress(proc, segments)

    t0 = time.perf_counter()
    triage = triage_video(video_path)
    ms = int((time.perf_counter() - t0) * 1000)
    log.info("Triage: ratio=%.4f passed=%s in %d ms", triage.changed_ratio, triage.passed, ms)
    result = CheckResult(video_ref(video_path), triage.info)
    done({"name": "triage", "completed_at": now_iso(), "duration_ms": ms,
          "changed_ratio": round(triage.changed_ratio, 4), "threshold": TRIAGE_CHANGED_RATIO,
          "passed": triage.passed}, [])
    if not triage.passed:
        return result

    t0 = time.perf_counter()
    result.info, segments = detect_segments(video_path)
    ms = int((time.perf_counter() - t0) * 1000)
    log.info("Video: %s - %d motion segment(s) in %d ms", result.info, len(segments), ms)
    result.segments = segments
    done({"name": "opencv", "completed_at": now_iso(), "duration_ms": ms}, segments)
    if not segments:
        return result

    t0 = time.perf_counter()
    fps = result.info.get("fps") or 25.0
    frames = read_frames(video_path, keyframe_indices(segments, fps))
    detections = detect_moving_objects(segments, frames, fps, result.info)
    classes: set[str] = set()
    for seg in segments:
        seg.detections = detections.get(seg.index, [])
        classes.update(d["class"] for d in seg.detections)
        # The centroid direction only means something with a real moving
        # subject; without one it tracks noise.
        if not seg.detections:
            seg.direction = "unknown"
            seg.centroid_start = seg.centroid_end = None
    ms = int((time.perf_counter() - t0) * 1000)
    log.info("YOLO detected classes %s in %d ms", sorted(classes), ms)
    done(yolo_verdict({"name": "yolo", "model": YOLO_MODEL_NAME, "completed_at": now_iso(), "duration_ms": ms,
                       "classes_detected": sorted(classes)}, required_classes), segments)
    return result
