"""The part of a clip worth keeping, and cutting the clip down to it.

Focus = while people move on screen (a YOLO scan over the whole clip), so a
car passing before anyone arrives, or after they leave, is cut. Clips without
a moving person fall back to the span of the OpenCV motion segments. The cut
keeps TRIM_PAD_SEC before and after; nothing in between is removed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from blink_watcher.settings import get_settings
from blink_watcher.vision.ffmpeg import trim_video
from blink_watcher.vision.motion import MotionSegment
from blink_watcher.vision.tracks import Sample, moving_person_tracks, scan_objects
from blink_watcher.vision.video import video_info

_settings = get_settings()

log = logging.getLogger(__name__)

TRIM_PAD_SEC = _settings.trim_pad_sec  # keep this much quiet video before the first and after the last activity


@dataclass
class Focus:
    span: Optional[tuple[float, float]]  # (start_sec, end_sec); None = no activity at all
    source: Optional[str]                # "person" | "motion"
    samples: list[Sample] = field(default_factory=list)  # the YOLO scan, reused for frame sampling

    def shifted(self, offset: float, duration: float) -> "Focus":
        """The same focus on the clock of a cut starting at `offset` and lasting `duration`."""
        span = None
        if self.span:
            span = (round(max(0.0, self.span[0] - offset), 3), round(min(duration, self.span[1] - offset), 3))
        samples = [(round(t - offset, 3), dets) for t, dets in self.samples if 0 <= t - offset <= duration]
        return Focus(span, self.source, samples)


def find_focus(video_path: Path, info: dict, segments: list[MotionSegment]) -> Focus:
    """No motion segments: no scan, no focus."""
    fps = info.get("fps") or 25.0
    duration = float(info.get("duration_sec") or 0.0)
    samples = scan_objects(video_path, fps, 0.0, duration) if segments else []
    people = moving_person_tracks(samples, info.get("width") or 0, info.get("height") or 0)
    if people:
        return Focus((min(tr.first_t for tr in people), max(tr.last_t for tr in people)), "person", samples)
    if segments:
        return Focus((min(s.start_sec for s in segments), max(s.end_sec for s in segments)), "motion", samples)
    return Focus(None, None, samples)


def trim_to_focus(video_path: Path, dst: Path, info: dict, focus: Focus) -> dict:
    """Cut `video_path` to the focus span (padded) into `dst`. Returns the trim
    record; a time on the cut clip + its offset_sec = the time on the original."""
    duration = float(info.get("duration_sec") or 0.0)
    t0 = max(0.0, focus.span[0] - TRIM_PAD_SEC)
    t1 = min(duration, focus.span[1] + TRIM_PAD_SEC)
    trim_video(video_path, dst, t0, t1)
    trimmed = video_info(dst)
    log.info("Trimmed to %s presence: %.2fs -> %.2fs (cut %.2fs head, %.2fs tail) -> %s",
             focus.source, duration, trimmed["duration_sec"] or 0.0, t0, duration - t1, dst)
    return {
        "file": dst.name,
        "focus": focus.source,
        "offset_sec": round(t0, 3),
        "source_end_sec": round(t1, 3),
        "pad_sec": TRIM_PAD_SEC,
        "removed_head_sec": round(t0, 3),
        "removed_tail_sec": round(duration - t1, 3),
        "duration_sec": trimmed["duration_sec"],
    }
