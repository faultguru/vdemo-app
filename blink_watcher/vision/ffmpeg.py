"""Cutting a clip with ffmpeg (the imageio-ffmpeg binary, or ffmpeg on PATH)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from blink_watcher.settings import get_settings
from blink_watcher.vision.video import video_info

_settings = get_settings()

TRIM_CRF = _settings.trim_crf                        # x264 quality for the trimmed video (lower = better)
TRIM_PRESET = "veryfast"       # x264 speed/size trade-off
TRIM_MAX_RATE_RATIO = _settings.trim_max_rate_ratio  # cap the re-encode at 90% of the source's average bitrate, so the trimmed file is always smaller


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise RuntimeError("ffmpeg not found - `pip install imageio-ffmpeg` or put ffmpeg on PATH")


def trim_video(src: Path, dst: Path, start_sec: float, end_sec: float) -> None:
    """Write [start_sec, end_sec] of `src` to `dst`: video re-encoded (H.264) so
    the cut lands on the exact frame (camera keyframes can be seconds apart, so
    a stream copy would keep extra lead-in), audio stream copied. The bitrate
    cap keeps the result smaller than the source. `src` is only read."""
    if dst.resolve() == src.resolve():
        raise ValueError(f"refusing to overwrite the source video: {src}")
    duration = video_info(src)["duration_sec"] or 0.0
    src_kbps = src.stat().st_size * 8 / 1000 / max(0.1, duration)
    max_kbps = max(200, int(src_kbps * TRIM_MAX_RATE_RATIO))
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
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
