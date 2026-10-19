"""Incoming clips for the web UI's live view: a day's clips on their way through
the pipeline, read from the storage areas (nothing is written).

  queued    <PROCESSED>/inbox/<stamp>-<CAMERA>.metadata.json, no work folder yet
  analyze   the clip's work folder exists (<TEMP>/<date>/<Camera>/<stamp>/): triage, OpenCV, YOLO
  trim      <stamp>.check.json written: cutting the quiet head / tail
  frames    <stamp>.analysis.json written: sampling key frames for the describer
  describe  the describer's request files written (describe.REQUEST_FILES_GLOB): the LLM
            describing the clip
  save      its result written (describe.RESULT_FILE): moving the results to the processed area
  notify    <CHAT>/events-inbox/<stamp>-<CAMERA>.event.json: processed, the
            event processor is deciding on the notification
  ignored   <IGNORED>/<date>/<Camera>/<stamp>.metadata.json written in the last
            RECENT_SEC seconds: dropped (filter, triage, OpenCV, YOLO or the describer)

Each item carries tags from the clip's .metadata.json (the trigger and the camera
provider's own detections) and, once the clip is analyzed, the classes YOLO found
(<stamp>.check.json); a person seen by either is one "person" tag naming its sources.

A clip leaves this list when its event reaches chat/<date>/events.json — the
day's events list the web UI already shows. Item ids are the event ids
(<stamp>-<CAMERA>), so the page can tell when a clip has arrived there.
"""
from __future__ import annotations

import json
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

from blink_watcher.chat.store import EVENTS_INBOX
from blink_watcher.core.storage import (
    CHAT, IGNORED_VIDEOS, PROCESSED_VIDEOS, TEMP_VIDEOS, camera_folder, clip_name, make_storage,
)
from blink_watcher.describe import REQUEST_FILES_GLOB, RESULT_FILE
from blink_watcher.settings import Settings

RECENT_SEC = 180        # an ignored clip is reported this long after it was dropped
STALLED_SEC = 180       # a work folder unchanged this long: failed, retried later
STEPS = ["queued", "analyze", "trim", "frames", "describe", "save", "notify"]
LABELS = {
    "queued": ("QUEUED", "waiting for the video processor"),
    "analyze": ("ANALYZING", "triage · OpenCV motion · YOLO person check"),
    "trim": ("TRIMMING", "cutting to the person activity"),
    "frames": ("SAMPLING", "extracting key frames for the describer"),
    "describe": ("DESCRIBING", "the LLM is analyzing the frames"),
    "save": ("SAVING", "moving the results to the processed area"),
    "notify": ("PROCESSED", "awaiting the notification decision"),
    "ignored": ("DISCARDED", ""),
}


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _video(block: Optional[dict]) -> Optional[str]:
    if block and block.get("folder") and block.get("video_file"):
        return f"{block['folder']}/{block['video_file']}"
    return None


def _work_stage(work: Path, stamp: str) -> tuple[str, Optional[float]]:
    """The processing step a work folder shows, and when it last changed (epoch s)."""
    if not work.is_dir():
        return "queued", None
    files = {p.name: p.stat().st_mtime for p in work.iterdir() if p.is_file()}
    changed = max(files.values(), default=work.stat().st_mtime)
    if RESULT_FILE in files or f"{stamp}.event.json" in files:
        return "save", changed
    if any(fnmatch(n, REQUEST_FILES_GLOB) for n in files):
        return "describe", changed
    if f"{stamp}.analysis.json" in files:
        return "frames", changed
    if f"{stamp}.check.json" in files:
        return "trim", changed
    return "analyze", changed


def _yolo_classes(work: Path, stamp: str) -> Optional[list[str]]:
    """Classes YOLO detected in the clip; None before YOLO has run."""
    check = _read(work / f"{stamp}.check.json") if work.is_dir() else None
    for proc in (check or {}).get("processors") or []:
        if proc.get("name") == "yolo":
            return list(proc.get("classes_detected") or [])
    return None


def _tags(md: dict, yolo: Optional[list[str]] = None) -> list[dict]:
    """[{text, kind, src}]: person first (kind "person"), then the other detections
    (kind "label"), then the trigger (kind "trigger")."""
    orig = md.get("original") or {}
    found: dict[str, list[str]] = {}
    for c in orig.get("provider_detections") or []:
        found.setdefault(str(c).lower(), []).append(orig.get("provider") or "camera")
    for c in yolo or []:
        found.setdefault(str(c).lower(), []).append("yolo")
    tags = [{"text": c, "kind": "person" if c == "person" else "label", "src": srcs}
            for c, srcs in sorted(found.items(), key=lambda kv: kv[0] != "person")]
    if orig.get("trigger"):
        tags.append({"text": str(orig["trigger"]), "kind": "trigger", "src": [orig.get("provider") or "camera"]})
    return tags


def _item(stage: str, md: dict, camera: str, stamp: str, **extra) -> dict:
    label, detail = LABELS[stage]
    return {
        "id": clip_name(camera, stamp),
        "camera": camera,
        "recorded_at": md.get("recorded_at"),
        "stage": stage,
        "step": STEPS.index(stage) if stage in STEPS else None,
        "steps": STEPS,
        "label": label,
        "detail": detail,
        "raw": _video(md.get("original")),
        "tags": _tags(md, extra.pop("yolo", None)),
        **extra,
    }


def incoming(settings: Settings, day: str, now: Optional[float] = None) -> list[dict]:
    """The day's clips not yet in its events list, newest recording first."""
    now = now or time.time()
    # globbed straight in the local folders: a few stat() calls per poll
    temp = make_storage(settings, TEMP_VIDEOS).local_path("")
    items: dict[str, dict] = {}

    # queued / processing — the video processor's inbox
    for f in sorted(make_storage(settings, PROCESSED_VIDEOS).local_path("inbox").glob(f"{day}-*.metadata.json")):
        md = _read(f)
        if not md or not md.get("camera"):
            continue
        camera, video = md["camera"], (md.get("original") or {}).get("video_file") or ""
        stamp = Path(video).stem or f.name.split(".")[0][:19]
        work = temp / day / camera_folder(camera) / stamp
        stage, changed = _work_stage(work, stamp)
        queued = f.stat().st_mtime
        it = _item(stage, md, camera, stamp, yolo=_yolo_classes(work, stamp), queued_ms=int(queued * 1000),
                   since_ms=int((changed or queued) * 1000))
        if changed and now - changed > STALLED_SEC:
            it["stalled"] = True
            it["detail"] = "no progress for a while — failed, will be retried"
        items[it["id"]] = it

    # processed — the event processor's inbox
    for f in sorted(make_storage(settings, CHAT).local_path(EVENTS_INBOX).glob(f"{day}-*.event.json")):
        ev = _read(f)
        if not ev or not ev.get("camera"):
            continue
        stamp = Path((ev.get("original") or {}).get("video_file") or ev.get("video") or "").stem
        md = {"recorded_at": ev.get("recorded_at"), "original": ev.get("original")}
        mtime = f.stat().st_mtime
        it = _item("notify", md, ev["camera"], stamp, queued_ms=int(mtime * 1000), since_ms=int(mtime * 1000),
                   processed=_video(ev.get("processed")) or ev.get("video"), thumbnail=ev.get("thumbnail"),
                   description=(ev.get("summary") or {}).get("description"),
                   threat=((ev.get("summary") or {}).get("threat") or {}).get("level"))
        it["id"] = ev.get("event_id") or it["id"]
        items[it["id"]] = it

    # recently dropped
    for f in make_storage(settings, IGNORED_VIDEOS).local_path(day).glob("*/*.metadata.json"):
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        if now - mtime > RECENT_SEC:
            continue
        md = _read(f)
        if not md or not md.get("camera"):
            continue
        ign = md.get("ignored") or {}
        if ign.get("by") == "filter":
            continue  # skipped by fetch (camera filter): never was in the pipeline
        stamp = f.name.split(".")[0]
        it = _item("ignored", md, md["camera"], stamp, since_ms=int(mtime * 1000),
                   ignored={"by": ign.get("by"), "reason": ign.get("reason")})
        it["detail"] = f"{ign.get('by') or 'pipeline'}: {ign.get('reason') or 'not important'}"
        items.setdefault(it["id"], it)

    return sorted(items.values(), key=lambda it: (it.get("recorded_at") or "", it["id"]), reverse=True)
