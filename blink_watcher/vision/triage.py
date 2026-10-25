"""Frame-change triage: the cheap first look at a clip. A few frames spread
over the clip are decoded at thumbnail size and compared pairwise; if no pair
differs noticeably, nothing happens in the clip and the expensive steps are
skipped."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from blink_watcher.settings import get_settings
from blink_watcher.vision.motion import FLASH_MEAN_DELTA
from blink_watcher.vision.video import capture_info, open_video

_settings = get_settings()

TRIAGE_SAMPLES = _settings.triage_samples              # evenly spaced frames decoded
TRIAGE_MAX_SIDE = _settings.triage_max_side            # downscale to this - absdiff at tiny resolution is near-free
TRIAGE_PIXEL_DELTA = _settings.triage_pixel_delta      # per-pixel intensity delta that counts as "changed"
TRIAGE_CHANGED_RATIO = _settings.triage_changed_ratio  # changed fraction of pixels between any two samples that counts as real change (0.5%)


@dataclass
class Triage:
    info: dict             # video info (video.capture_info)
    passed: bool           # something changes
    changed_ratio: float   # the largest changed-pixel fraction over the sample pairs


def triage_video(video_path: Path) -> Triage:
    """Pairs whose global brightness differs are exposure flashes and don't
    count. A clip whose frames can't be sampled passes (the full pipeline decides)."""
    cap = open_video(video_path)
    try:
        info = capture_info(cap)
        frame_count, width, height = info["frame_count"], info["width"], info["height"]
        if frame_count <= 0:
            return Triage(info, True, 0.0)

        # Skip the very start: the first frames can be an exposure flash
        # before auto-exposure settles.
        start = max(1, frame_count // 20)
        step = max(1, (frame_count - start - 1) // max(1, TRIAGE_SAMPLES - 1))
        targets = [min(frame_count - 1, start + i * step) for i in range(TRIAGE_SAMPLES)]

        scale = min(1.0, TRIAGE_MAX_SIDE / max(width, height)) if max(width, height) else 1.0
        grays: list[np.ndarray] = []
        for t in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, frame = cap.read()
            if not ok:
                continue
            if scale < 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()

    if len(grays) < 2:
        return Triage(info, True, 0.0)

    # Every pair: a subject visible in a single mid-clip frame still stands
    # out against the frames before and after it.
    means = [float(g.mean()) for g in grays]
    max_ratio = 0.0
    for i in range(len(grays)):
        for j in range(i + 1, len(grays)):
            if abs(means[i] - means[j]) >= FLASH_MEAN_DELTA:
                continue
            changed = float((cv2.absdiff(grays[i], grays[j]) >= TRIAGE_PIXEL_DELTA).mean())
            max_ratio = max(max_ratio, changed)
    return Triage(info, max_ratio >= TRIAGE_CHANGED_RATIO, max_ratio)
