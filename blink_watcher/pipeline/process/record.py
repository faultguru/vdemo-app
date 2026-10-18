"""What the video processor writes about a clip: the processed .metadata.json
body (with its "llm" block), the "ignored" record, and the .event.json."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from blink_watcher.core.storage import file_link
from blink_watcher.describe.describer import Description
from blink_watcher.pipeline.events import event_entry
from blink_watcher.pipeline.process.clip import Areas, Clip
from blink_watcher.vision.motion import MotionSegment

# The describer's answer counts as empty with one of these descriptions (and
# no people, vehicles or actions).
EMPTY_DESCRIPTIONS = ("", "no motion", "no motion detected")


def ignored(by: str, reason: str) -> dict:
    return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "by": by, "reason": reason}


def ignore_decision(record: dict, llm: Optional[dict], keep_vehicles: bool = False) -> Optional[dict]:
    """The "ignored" record if the pipeline judged the clip unimportant, else None:
    by the LLM's answer when there is one, else by the first failed check. The
    LLM never discards a clip with a person in its answer, nor one with a vehicle
    when `keep_vehicles` (the camera's filter is person_or_vehicle); what it
    judged not to be a real person (a decoration, statue, poster) is not listed."""
    if llm:
        s = llm.get("summary") or {}
        if ((s.get("threat") or {}).get("level") or "none") != "none":
            return None  # suspicious activity / threat: always kept, whatever "ignore" says
        if s.get("people") or (keep_vehicles and s.get("vehicles")):
            return None  # a person (or a vehicle, per the camera's filter) in the clip: always kept
        return ignored("llm", s.get("description") or "nothing of importance") if s.get("ignore") else None
    procs = {p.get("name"): p for p in record.get("processors") or []}
    if "triage" in procs and not procs["triage"].get("passed"):
        return ignored("triage", "nothing changes between sampled frames")
    if not record.get("motion_segments"):
        return ignored("opencv", "no motion segments")
    yolo = procs.get("yolo") or {}
    if not yolo.get("passed"):
        found = ", ".join(yolo.get("classes_detected") or []) or "nothing"
        if yolo.get("required_classes"):
            return ignored("yolo", f"no {' / '.join(yolo['required_classes'])} detected moving (found: {found})")
        return ignored("yolo", "no objects detected in the motion")
    return None


def shift_segments(segments: list[MotionSegment], offset: float, duration: float) -> list[MotionSegment]:
    """Motion segments on the original clip's clock -> the trimmed clip's
    clock; segments entirely outside the kept range are dropped."""
    out = []
    for s in segments:
        start, end = s.start_sec - offset, s.end_sec - offset
        if end <= 0 or start >= duration:
            continue
        start, end = max(0.0, start), min(duration, end)
        out.append(MotionSegment.from_record({
            **s.to_record(),
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(end - start, 3),
            "keyframe_times_sec": [round(t - offset, 3) for t in s.keyframe_times_sec if 0 <= t - offset <= duration],
        }))
    return out


def mark_empty_ignored(description: Description) -> bool:
    """An answer with nothing in it (no description, people, vehicles or
    actions) is noise: set its ignore flag. True if it was set."""
    s = description.analysis.summary
    if (s.description or "").strip().lower().rstrip(".") in EMPTY_DESCRIPTIONS \
            and not s.people and not s.vehicles and not s.actions:
        s.ignore = True
        return True
    return False


def llm_block(description: Description, segments: list[MotionSegment], completed_at: str, duration_ms: int) -> dict:
    """The describer's answer + call details: the "llm" block of the metadata."""
    analysis, response = description.analysis, description.response
    usage = response.usage
    by_index = {s.index: s for s in segments}
    events = []
    for ev in analysis.events:
        seg = by_index.get(ev.segment_index)
        events.append({
            "segment_index": ev.segment_index,
            "start_sec": seg.start_sec if seg else None,
            "end_sec": seg.end_sec if seg else None,
            "object": ev.object,
            "details": ev.details,
            "description": ev.description,
        })
    return {
        "provider": response.provider,
        "model": response.model,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "frames_sent": description.frames_sent,
        "stop_reason": response.stop_reason,
        "retries": response.retries,
        "usage": {
            "input_tokens": usage.input_tokens,
            "cache_write_tokens": usage.cache_creation_input_tokens,
            "cache_read_tokens": usage.cache_read_input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": response.cost_usd,
        },
        "summary": analysis.summary.model_dump(),
        "events": events,
        "thumbnail_sec": analysis.thumbnail_sec if analysis.thumbnail_sec >= 0 else None,
        "important_frames_sec": analysis.important_frames_sec,
    }


def build_record(md: dict, info: dict, segments: list[MotionSegment], processors: list[dict],
                 llm: Optional[dict] = None, processed: Optional[dict] = None) -> dict:
    """The .metadata.json body: the queued metadata, the processed-file link,
    the analysis (info, motion_segments), processors in run order (sequence
    1..N) and the llm block."""
    return {
        **md,
        **({"processed": processed} if processed else {}),
        "info": info,
        "motion_segments": [s.to_record() for s in segments],
        "processors": [
            {"sequence": i, **{k: v for k, v in p.items() if k != "sequence"}}
            for i, p in enumerate(processors, start=1)
        ],
        **({"llm": llm} if llm else {}),
    }


def build_event(areas: Areas, clip: Clip, record: dict, thumbnail: bool, **extra) -> dict:
    """The clip's event in the events index format (pipeline.events), pointing
    at the processed video and metadata (paths relative to STORAGE_ROOT, e.g.
    videos.processed/<date>/<camera>/<stamp>.mp4; the video is never copied),
    plus "original" / "processed" video links as in the .metadata.json, the
    security mode it was processed in and `extra` fields (a fast clip's "fast" and
    "notification")."""
    entry = event_entry(
        clip.event_id,
        f"{areas.processed.folder}/{clip.key}",
        f"{areas.processed.folder}/{clip.metadata_key}",
        (record.get("llm") or {}).get("summary") or {},
        thumbnail=f"{areas.processed.folder}/{clip.thumbnail_key}" if thumbnail else None,
        recorded_at=record.get("recorded_at"),
    )
    links = {
        "original": file_link(areas.raw.folder, clip.key),
        "processed": file_link(areas.processed.folder, clip.key),
    }
    # Camera right after the event id; links right after the video / metadata paths.
    items = list(entry.items())
    at = list(entry).index("metadata") + 1
    event = dict(items[:1] + [("camera", clip.camera)] + items[1:at] + list(links.items()) + items[at:])
    if record.get("security_mode"):
        event["security_mode"] = record["security_mode"]
    return {**event, **extra}


def fast_summary() -> dict:
    """The summary of a clip saved without analysis (home security mode, indoor camera)."""
    return {"description": "Indoor clip while the residents were home - not analysed",
            "labels": ["residents home"]}
