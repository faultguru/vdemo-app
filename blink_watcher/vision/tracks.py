"""Object tracks over a whole clip: a YOLO scan at SCAN_FPS and a greedy IoU
tracker that links the detections into tracks — enough to tell two or three
people on a porch apart, and to tell a person walking from a person-shaped
decoration that never moves."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from blink_watcher.settings import get_settings
from blink_watcher.vision.video import open_video
from blink_watcher.vision.yolo import get_yolo

_settings = get_settings()

SCAN_FPS = _settings.track_scan_fps                    # YOLO scan rate
SCAN_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat")
DETECT_CONF = _settings.track_detect_conf              # min YOLO confidence for a box
YOLO_BATCH = _settings.track_yolo_batch                # frames per YOLO predict call
TRACK_IOU = _settings.track_iou                        # min IoU to continue a track
TRACK_MAX_GAP_SEC = _settings.track_max_gap_sec        # a track ends if unseen this long
MIN_TRACK_SAMPLES = _settings.track_min_samples        # ignore tracks shorter than this (flicker)
PERSON_MOVE_RATIO = _settings.track_person_move_ratio  # a person track counts only if its center moves >=5% of the frame's long side - decorations, statues and posters don't
OBJECT_MOVE_RATIO = _settings.track_object_move_ratio  # a vehicle / animal track counts only if its center moves >=3% of the frame's long side - parked cars don't

Box = list[float]                         # x0, y0, x1, y1
Detection = tuple[Box, float, str]        # box, confidence, class name
Sample = tuple[float, list[Detection]]    # (t, detections in the frame at t)


@dataclass
class Track:
    id: int
    hits: list[tuple[float, Box, float]] = field(default_factory=list)  # (t, box, confidence)

    @property
    def first_t(self) -> float:
        return self.hits[0][0]

    @property
    def last_t(self) -> float:
        return self.hits[-1][0]

    @property
    def last_box(self) -> Box:
        return self.hits[-1][1]

    def moved(self, min_shift: float) -> bool:
        """The box center moved at least `min_shift` px over the track."""
        cxs = [(b[0] + b[2]) / 2 for _, b, _ in self.hits]
        cys = [(b[1] + b[3]) / 2 for _, b, _ in self.hits]
        return ((max(cxs) - min(cxs)) ** 2 + (max(cys) - min(cys)) ** 2) ** 0.5 >= min_shift


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def scan_objects(video_path: Path, fps: float, start_sec: float, end_sec: float) -> list[Sample]:
    """Decode [start, end] sequentially and run YOLO (SCAN_CLASSES only) every 1/SCAN_FPS seconds."""
    model = get_yolo()
    class_ids = [i for i, n in model.names.items() if n in SCAN_CLASSES]
    start_f = max(0, int(round(start_sec * fps)))
    end_f = int(round(end_sec * fps))
    step = max(1, int(round(fps / SCAN_FPS)))

    samples: list[Sample] = []
    batch_t: list[float] = []
    batch_f: list[np.ndarray] = []

    def flush() -> None:
        if not batch_f:
            return
        for t, r in zip(batch_t, model.predict(batch_f, conf=DETECT_CONF, classes=class_ids, verbose=False)):
            dets = [([float(v) for v in box.xyxy[0].tolist()], float(box.conf.item()), model.names[int(box.cls.item())])
                    for box in (r.boxes if r.boxes is not None else [])]
            samples.append((t, dets))
        batch_t.clear()
        batch_f.clear()

    cap = open_video(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for idx in range(start_f, end_f + 1):
            if (idx - start_f) % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                batch_t.append(round(idx / fps, 3))
                batch_f.append(frame)
                if len(batch_f) >= YOLO_BATCH:
                    flush()
            elif not cap.grab():
                break
        flush()
    finally:
        cap.release()
    return samples


def build_tracks(samples: list[Sample], classes: set[str]) -> list[Track]:
    """Greedy IoU tracker over the detections of `classes`."""
    tracks: list[Track] = []
    for t, all_dets in samples:
        dets = [(box, conf) for box, conf, name in all_dets if name in classes]
        matched: set[int] = set()
        for box, conf in sorted(dets, key=lambda d: -d[1]):
            best, best_iou = None, TRACK_IOU
            for tr in tracks:
                if tr.id in matched or t - tr.last_t > TRACK_MAX_GAP_SEC:
                    continue
                overlap = iou(box, tr.last_box)
                if overlap >= best_iou:
                    best, best_iou = tr, overlap
            if best is None:
                best = Track(id=len(tracks))
                tracks.append(best)
            best.hits.append((t, box, conf))
            matched.add(best.id)
    return [tr for tr in tracks if len(tr.hits) >= MIN_TRACK_SAMPLES]


def moving_person_tracks(samples: list[Sample], frame_w: int, frame_h: int) -> list[Track]:
    """Person tracks that actually move. A person-shaped object that stays put
    (a door decoration, a statue, a poster) is detected as a person by YOLO -
    and by the camera's own detector - but never moves."""
    min_shift = max(frame_w, frame_h) * PERSON_MOVE_RATIO
    return [tr for tr in build_tracks(samples, {"person"}) if tr.moved(min_shift)]
