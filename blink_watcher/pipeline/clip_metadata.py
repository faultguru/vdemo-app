"""The clip record (.metadata.json) written next to each downloaded clip — our
own, provider-neutral format:

  {
    "version": 1,
    "camera": "Front Door",
    "date": "2026-10-17",                           # local date of recorded_at
    "recorded_at": "2026-10-17T00:03:18-07:00",     # start of the recording, local time with UTC offset
    "original": {                                   # the clip as the provider delivered it
      "provider": "blink",
      "provider_clip_id": "17524422150",
      "provider_time": "2026-10-17T00:03:48-07:00", # the provider's own clip time (Blink: end of the recording)
      "duration_sec": 30.061,
      "trigger": "motion",                          # motion | snapshot | live_view | provider's own value
      "provider_detections": ["vehicle"],           # provider's own object labels; [] if none
      "folder": "videos.raw/2026-10-17/Front_Door", # relative to STORAGE_ROOT (starts with the area folder)
      "video_file": "2026-10-17-00-03-18.mp4"
    }
  }

A clip not downloaded because of its camera's filter (CAMERA_FILTERS) also gets
  "download": {"status": "filtered", "filter": "person", "reason": ..., "checked_at": ...}

The same file grows as the clip moves through the pipeline: the video
processor adds "processed": {"folder", "video_file"} (the trimmed clip) and
its analysis results.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from blink_watcher.core.storage import file_link
from blink_watcher.providers import ProviderClip

METADATA_VERSION = 1


def clip_metadata(clip: ProviderClip, tz: ZoneInfo, area_folder: str, video_key: str) -> dict:
    """The record of a clip stored (or to be stored) at `video_key` in the area `area_folder`."""
    local = clip.recorded_at.astimezone(tz)
    return {
        "version": METADATA_VERSION,
        "camera": clip.camera.strip(),
        "date": f"{local:%Y-%m-%d}",
        "recorded_at": local.isoformat(),
        "original": {
            "provider": clip.provider,
            "provider_clip_id": clip.provider_clip_id,
            **({"provider_time": clip.provider_time.astimezone(tz).isoformat()} if clip.provider_time else {}),
            "duration_sec": clip.duration_sec,
            "trigger": clip.trigger,
            "provider_detections": list(clip.detections),
            **file_link(area_folder, video_key),
        },
    }


def filtered_download(filter_name: str, detections: list[str], tz: ZoneInfo) -> dict:
    """The "download" entry of a clip its camera's filter rejected (video not downloaded)."""
    return {
        "status": "filtered",
        "filter": filter_name,
        "reason": f"filter {filter_name}: provider detections "
                  f"{detections or 'none'} do not match - video not downloaded",
        "checked_at": datetime.now(tz).isoformat(timespec="seconds"),
    }


def filtered_by(md: dict) -> Optional[str]:
    """The filter that rejected the clip, if its record is marked filtered."""
    download = md.get("download") or {}
    return download.get("filter") if download.get("status") == "filtered" else None
