"""The event record and the notification prompt.

event_entry() builds an event as stored in the events index
(chat/<date>/events.json) and written as <stamp>.event.json by the video
processor (pipeline.process); notification_prompt() is the per-event input of
the chat agent's notify role (prompts/chat_notify.system.txt).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

from blink_watcher.settings import get_settings


DEFAULT_TIMEZONE = "America/Los_Angeles"  # when the settings can't be loaded


def _recorded_at(event_id: str) -> str:
    """The ISO time in an event id (<YYYY-MM-DD-HH-MM-SS>-...) in the
    configured timezone; now if the id has none."""
    try:
        dt = datetime.strptime(event_id[:19], "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        return datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        tz_name = get_settings().timezone
    except SystemExit:
        tz_name = DEFAULT_TIMEZONE
    return dt.replace(tzinfo=ZoneInfo(tz_name)).isoformat(timespec="seconds")


def _trim_summary(summary: dict) -> dict:
    """The describer's summary as stored with the event: without "ignore",
    empty threat / vehicles dropped, people_types and all labels up front."""
    out = {k: v for k, v in summary.items() if k != "ignore"}
    if ((out.get("threat") or {}).get("level") or "none") == "none":
        out.pop("threat", None)  # only kept when there is something to report
    if out.get("vehicles") == []:
        out.pop("vehicles", None)
    people_types = list(dict.fromkeys(p["type"] for p in out.get("people") or [] if p.get("type")))
    # All free-form labels (clip, people, vehicles) in one list, next to people_types.
    labels = list(dict.fromkeys(
        [*(out.get("labels") or []),
         *(lbl for p in out.get("people") or [] for lbl in p.get("labels") or []),
         *(lbl for v in out.get("vehicles") or [] for lbl in v.get("labels") or [])]))
    reordered: dict = {"description": out.get("description", "")}
    if people_types:
        reordered["people_types"] = people_types
    if labels:
        reordered["labels"] = labels
    for k, v in out.items():
        if k != "description":
            reordered[k] = v
    return reordered


def event_entry(event_id: str, video: str, metadata: str, summary: dict,
                thumbnail: Optional[str] = None, recorded_at: Optional[str] = None) -> dict:
    """One event in the events index format (also written as <stamp>.event.json
    by pipeline.process). video / metadata / thumbnail are stored as given."""
    return {
        "event_id": event_id,
        "recorded_at": recorded_at or _recorded_at(event_id),
        "video": video,
        "metadata": metadata,
        "thumbnail": thumbnail,
        "summary": _trim_summary(summary or {}),
    }


def notification_prompt(recorded_at: str, summary: dict, offer_media: bool, camera: Optional[str] = None,
                        security_mode: Optional[str] = None, auto_away: bool = False,
                        vehicle_clips: Sequence[str] = ()) -> str:
    """The per-event input of the notify role: offer_media flag, local time,
    camera, the security mode the clip was processed in, whether auto-away is
    on (the household_vehicle answer is wanted) with the camera's nearby
    vehicle-only clips (nearby_vehicle_clips), summary."""
    try:
        dt = datetime.fromisoformat(recorded_at)
        local_time = dt.strftime("%I:%M%p").lstrip("0").lower()
        local_time = f"{local_time} on {dt.strftime('%A, %Y-%m-%d')}"
    except ValueError:
        local_time = recorded_at
    return "\n".join([
        f"offer_media: {'true' if offer_media else 'false'}",
        f"We have a new event received at {local_time} ({recorded_at})"
        + (f" from the {camera} camera" if camera else "")
        + (f", while the security mode was {security_mode.upper()}" if security_mode else "") + ".",
        *(["Auto-away: on (fill household_vehicle)"] if auto_away else []),
        *([f"Vehicle-only clips of this camera close to this one (not downloaded, no description): "
           f"{'; '.join(vehicle_clips)}"] if auto_away and vehicle_clips else []),
        "Summary:",
        json.dumps(summary, indent=2),
    ])


def nearby_vehicle_clips(settings, camera: Optional[str], recorded_at: str) -> list[str]:
    """Clips of the same camera starting within AUTO_AWAY_CLIP_WINDOW_SEC of
    the event that the download filter skipped and the camera tagged only
    "vehicle", as "18:10:26, 60 s long, 71 s before this clip". A car pulling
    in or driving off is often such a clip, the person getting out or in a
    clip of its own."""
    from blink_watcher.core.storage import RAW_VIDEOS, camera_folder, make_storage
    window = settings.auto_away_clip_window_sec
    try:
        at = datetime.fromisoformat(recorded_at)
    except ValueError:
        return []
    if not camera or window <= 0:
        return []
    raw = make_storage(settings, RAW_VIDEOS)
    found = []
    for key in sorted(raw.list(f"{at:%Y-%m-%d}/{camera_folder(camera)}/")):
        if not key.endswith(".metadata.json"):
            continue
        try:
            meta = raw.read_json(key)
            start = datetime.fromisoformat(meta["recorded_at"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        original = meta.get("original") or {}
        tags = set(original.get("provider_detections") or [])
        offset = (start - at).total_seconds()
        if ((meta.get("download") or {}).get("status") != "filtered" or "vehicle" not in tags
                or "person" in tags or abs(offset) > window):
            continue
        length = original.get("duration_sec")
        found.append(f"{start:%H:%M:%S}" + (f", {length:.0f} s long" if length else "")
                     + f", {abs(offset):.0f} s {'before' if offset < 0 else 'after'} this clip")
    return found
