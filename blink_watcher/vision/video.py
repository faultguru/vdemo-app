"""Reading video files with OpenCV: stream info, frames at given times or
frame indices, and frames as JPEG."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from blink_watcher.settings import get_settings

_settings = get_settings()

DEFAULT_FPS = 25.0      # when the container reports none (broken headers)
JPEG_MAX_SIDE = _settings.thumbnail_max_side  # thumbnails
JPEG_QUALITY = _settings.thumbnail_quality


class VideoError(RuntimeError):
    pass


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"could not open video: {path}")
    return cap


def capture_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    return fps if fps > 0 else DEFAULT_FPS


def capture_info(cap: cv2.VideoCapture) -> dict:
    """{fps, frame_count, width, height, duration_sec} — the "info" block of the
    metadata files; duration_sec is None when the frame count is unknown."""
    fps = capture_fps(cap)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return {
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "duration_sec": round(frame_count / fps, 3) if frame_count else None,
    }


def video_info(path: Path) -> dict:
    cap = open_video(path)
    try:
        return capture_info(cap)
    finally:
        cap.release()


def read_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """{frame index: BGR frame} by seeking; unreadable indices are left out."""
    out: dict[int, np.ndarray] = {}
    if not indices:
        return out
    cap = open_video(path)
    try:
        for i in sorted(set(indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                out[i] = frame
    finally:
        cap.release()
    return out


def read_frames_at(path: Path, times_sec: list[float]) -> dict[float, np.ndarray]:
    """{timestamp: BGR frame} for each requested time (seconds from the start)."""
    if not times_sec:
        return {}
    cap = open_video(path)
    fps = capture_fps(cap)
    out: dict[float, np.ndarray] = {}
    try:
        for t in sorted(set(times_sec)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if ok:
                out[t] = frame
    finally:
        cap.release()
    return out


def encode_jpeg(frame: np.ndarray, max_side: int = JPEG_MAX_SIDE, quality: int = JPEG_QUALITY) -> bytes:
    """A BGR frame as JPEG, downscaled to at most `max_side` px on its longer side."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoError("JPEG encode failed")
    return buf.tobytes()


def video_ref(path: Path) -> str:
    """The clip's path below its storage area folder ("videos" / "videos.*"),
    e.g. "2026-10-17/Front_Door/2026-10-17-00-03-18.mp4"; the bare file name
    when it is not inside one."""
    parts = path.resolve().parts
    for i in range(len(parts) - 1, -1, -1):
        name = parts[i].lower()
        if name == "videos" or name.startswith("videos."):
            return "/".join(parts[i + 1:])
    return path.name
