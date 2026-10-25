"""The local web console."""
from __future__ import annotations

from pydantic_settings import BaseSettings


class WebUISettings(BaseSettings):
    webui_host: str = "127.0.0.1"
    webui_port: int = 9090
    webui_poll_sec: float = 3.0                 # how often the live view checks for new events
    webui_chat_lookback_days: int = 14          # days the console chat covers by default
    # Live monitor: hours snapshots are kept, and what the analyst sees of them.
    webui_live_keep_hours: float = 6.0
    webui_live_ai_frames: int = 3
    webui_live_ai_window_sec: float = 120.0
    # Tuning (.env.example documents each option):
    webui_max_upload_mb: int = 200
    webui_chat_transcript_max: int = 100
    webui_chat_video_frames: int = 4
    webui_incoming_recent_sec: int = 180
    webui_incoming_stalled_sec: int = 180
    webui_camera_fallback_days: int = 14
    live_viewer_timeout_sec: float = 15.0
    live_round_pause_sec: float = 1.0
    live_error_pause_sec: float = 10.0
    live_prune_every_sec: float = 600.0
    live_stream_fps: int = 8
    live_stream_snapshot_sec: float = 5.0
    live_stream_idle_sec: float = 8.0
    live_stream_stall_sec: float = 25.0
