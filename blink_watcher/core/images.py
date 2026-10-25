"""Picture helpers shared by the chat agent, the live monitor and the web UI."""
from __future__ import annotations
from blink_watcher.settings import get_settings

_settings = get_settings()

IMAGE_MAX_SIDE = _settings.llm_image_max_side  # pictures shown to an LLM are downscaled to this (tokens)


class ImageError(ValueError):
    pass


def fit_jpeg(data: bytes, max_side: int = IMAGE_MAX_SIDE, quality: int = 85) -> bytes:
    """Image bytes (JPEG / PNG / ...) as a JPEG at most `max_side` px on its longer side."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageError("not a readable picture")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageError("picture could not be encoded")
    return jpg.tobytes()
