"""Live monitor of the web UI: snapshot loop and MJPEG live view, on the camera-provider API."""
from __future__ import annotations

from blink_watcher.live.monitor import LiveMonitor
from blink_watcher.live.stream import LiveError, LiveStream

__all__ = ["LiveError", "LiveMonitor", "LiveStream"]
