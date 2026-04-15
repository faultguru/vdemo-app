"""CLI: copy a video into events/, analyze the copy, keep it only if not ignored.

Flow:
  1. Pick an event id like YYYY-MM-DD-HH-MM-SS-NN (NN starts at 00, increments
     on collision) from the source video's filename timestamp.
  2. Copy the source video into events/<eventid>/<eventid>.mp4 — source is
     never modified.
  3. Run the usual describe_video pipeline on the *copy*; metadata JSON,
     thumbnail, and keyframes all land next to it inside that folder.
  4. If summary.ignore is True, delete the whole events/<eventid>/ folder.
  5. Otherwise, append an entry to events/events.YYYY-MM-DD.json with links
     to the metadata, thumbnail, and video inside events/<eventid>/.

Usage:
    blink-event path/to/clip.mp4
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from blink_watcher.config import load_settings
from blink_watcher.describe_video import analyze

log = logging.getLogger("blink_watcher.event_log")

EVENTS_DIR = Path("events")
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")


def _base_event_id(video_path: Path) -> str:
    m = TIMESTAMP_RE.match(video_path.stem)
    if m:
        return m.group(1)
    return datetime.fromtimestamp(video_path.stat().st_mtime).strftime("%Y-%m-%d-%H-%M-%S")


def _allocate_event_video(events_dir: Path, base: str) -> tuple[str, Path, Path]:
    """Return (event_id, date_dir, video_copy_path) for the next free 2-digit
    sequence under events/<date>/. Date dir is created if missing."""
    date_dir = events_dir / base[:10]
    date_dir.mkdir(parents=True, exist_ok=True)
    for seq in range(100):
        event_id = f"{base}-{seq:02d}"
        video_copy = date_dir / f"{event_id}.mp4"
        if not video_copy.exists():
            return event_id, date_dir, video_copy
    raise RuntimeError(f"all 100 sequence slots for {base} are taken")


def _event_date(event_id: str) -> str:
    return event_id[:10]


def _recorded_at(event_id: str) -> str:
    """Derive ISO timestamp from the event_id (YYYY-MM-DD-HH-MM-SS-NN) in the
    configured timezone. Falls back to now() if the id doesn't parse."""
    try:
        dt = datetime.strptime(event_id[:19], "%Y-%m-%d-%H-%M-%S")
        tz_name = "America/Los_Angeles"
        try:
            tz_name = load_settings().timezone
        except SystemExit:
            pass
        return dt.replace(tzinfo=ZoneInfo(tz_name)).isoformat(timespec="seconds")
    except ValueError:
        return datetime.now().astimezone().isoformat(timespec="seconds")


def _rel_to_project(p: Path) -> str:
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def _trim_summary(summary: dict) -> dict:
    out = {k: v for k, v in summary.items() if k != "ignore"}
    if out.get("vehicles") == []:
        out.pop("vehicles", None)
    seen: list[str] = []
    for p in out.get("people") or []:
        t = p.get("type")
        if t and t not in seen:
            seen.append(t)
    people_types = seen
    reordered: dict = {"description": out.get("description", "")}
    if people_types:
        reordered["people_types"] = people_types
    for k, v in out.items():
        if k != "description":
            reordered[k] = v
    return reordered


def _append_index(event_id: str, metadata_path: Path, video_copy: Path, events_dir: Path) -> Path:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = _trim_summary(metadata.get("summary") or {})
    thumb_path = video_copy.with_suffix(".jpg")

    entry = {
        "event_id": event_id,
        "recorded_at": _recorded_at(event_id),
        "video": _rel_to_project(video_copy),
        "metadata": _rel_to_project(metadata_path),
        "thumbnail": _rel_to_project(thumb_path) if thumb_path.exists() else None,
        "summary": summary,
    }

    out = events_dir / f"events.{_event_date(event_id)}.json"
    existing: list[dict] = []
    if out.exists():
        try:
            loaded = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except json.JSONDecodeError:
            log.warning("could not parse existing %s — rewriting", out)
    existing = [e for e in existing if e.get("event_id") != event_id]
    existing.append(entry)
    existing.sort(key=lambda e: e.get("event_id") or "")
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    log.info("appended event %s → %s (%d entries)", event_id, out, len(existing))
    return out


OFFER_COOLDOWN_SECONDS = 3600  # only invite the user to request media once per hour


def _should_offer_media(events_dir: Path) -> bool:
    state_file = events_dir / ".notify_state.json"
    now = datetime.now().timestamp()
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        last = float(state.get("last_offer_ts") or 0)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        last = 0.0
    if now - last < OFFER_COOLDOWN_SECONDS:
        return False
    state_file.write_text(json.dumps({"last_offer_ts": now}), encoding="utf-8")
    return True


def _enqueue_outbox(
    event_id: str,
    summary: dict,
    events_dir: Path,
) -> None:
    """Drop a file in events/outbox/ for the chat bot to deliver to Telegram.

    File format matches what chat_bot._process_outbox_file() expects:
      {"event_id": "...", "prompt": "<raw facts>"}
    The chat bot derives the event thumbnail from the event_id.
    """
    if not (summary.get("description") or "").strip():
        return

    recorded_at = _recorded_at(event_id)
    try:
        dt = datetime.fromisoformat(recorded_at)
        local_time = dt.strftime("%I:%M%p").lstrip("0").lower()
        local_time = f"{local_time} on {dt.strftime('%A, %Y-%m-%d')}"
    except ValueError:
        local_time = recorded_at
    offer_media = "true" if _should_offer_media(events_dir) else "false"
    prompt_lines = [
        f"offer_media: {offer_media}",
        f"We have a new event received at {local_time} ({recorded_at}). Summary:",
        json.dumps(summary, indent=2),
    ]
    payload = {
        "event_id": event_id,
        "prompt": "\n".join(prompt_lines),
    }

    outbox_dir = events_dir / "outbox"
    outbox_dir.mkdir(parents=True, exist_ok=True)
    out_file = outbox_dir / f"{event_id}.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("outbox: queued %s", out_file)


def _discard_artifacts(video_copy: Path) -> None:
    """Remove the copied video and any sibling artifacts analyze() produced."""
    stem = video_copy.stem
    parent = video_copy.parent
    video_copy.unlink(missing_ok=True)
    (parent / f"{stem}.json").unlink(missing_ok=True)
    (parent / f"{stem}.jpg").unlink(missing_ok=True)
    shutil.rmtree(parent / f"{stem}.keyframes", ignore_errors=True)


def _reuse_source_artifacts(
    video_path: Path,
    events_dir: Path,
) -> Optional[Path]:
    """Token-saver fast path: if the source folder already has a metadata JSON
    (produced by a prior analyze() run), copy it + thumbnail + keyframes +
    video into the event folder without re-running the pipeline. Skips
    entirely when summary.ignore is True — no files are copied in that case.

    Returns the events index path on success, or None if the fast path doesn't
    apply (no source metadata, parse error) so the caller can fall back.
    """
    source_json = video_path.with_suffix(".json")
    if not source_json.exists():
        return None
    try:
        metadata = json.loads(source_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.warning("token-saver: source metadata %s is not valid JSON (%s) — falling back", source_json, e)
        return None

    if (metadata.get("summary") or {}).get("ignore"):
        log.info("token-saver: %s has summary.ignore=True — skipping", video_path.name)
        return None

    base = _base_event_id(video_path)
    event_id, date_dir, video_copy = _allocate_event_video(events_dir, base)
    log.info("token-saver: event_id=%s  → %s (reusing %s)", event_id, video_copy, source_json.name)

    try:
        shutil.copy2(video_path, video_copy)
        metadata["video"] = video_copy.name  # rewrite so the saved json matches the new location
        metadata_copy = video_copy.with_suffix(".json")
        metadata_copy.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        source_thumb = video_path.with_suffix(".jpg")
        if source_thumb.exists():
            shutil.copy2(source_thumb, video_copy.with_suffix(".jpg"))

        source_kf = video_path.parent / f"{video_path.stem}.keyframes"
        if source_kf.is_dir():
            shutil.copytree(source_kf, video_copy.parent / f"{video_copy.stem}.keyframes")

        index_path = _append_index(event_id, metadata_copy, video_copy, events_dir)
        _enqueue_outbox(event_id, _trim_summary(metadata.get("summary") or {}), events_dir)
        return index_path
    except BaseException:
        _discard_artifacts(video_copy)
        raise


def process_video(
    video_path: Path,
    events_dir: Path = EVENTS_DIR,
    motion_only: bool = False,
    token_saver: bool = False,
) -> Optional[Path]:
    events_dir.mkdir(parents=True, exist_ok=True)

    if token_saver:
        result = _reuse_source_artifacts(video_path, events_dir)
        if result is not None:
            return result
        # No reusable source metadata — fall through to full analyze.
        if not video_path.with_suffix(".json").exists():
            log.info("token-saver: no source metadata for %s — running full pipeline", video_path.name)

    base = _base_event_id(video_path)
    event_id, date_dir, video_copy = _allocate_event_video(events_dir, base)
    log.info("event_id=%s  → %s", event_id, video_copy)

    shutil.copy2(video_path, video_copy)
    try:
        metadata_path = analyze(video_copy, motion_only=motion_only)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("summary") or {}).get("ignore"):
            log.info("summary.ignore=True — discarding %s", event_id)
            _discard_artifacts(video_copy)
            return None
        index_path = _append_index(event_id, metadata_path, video_copy, events_dir)
        _enqueue_outbox(event_id, _trim_summary(metadata.get("summary") or {}), events_dir)
        return index_path
    except BaseException:
        _discard_artifacts(video_copy)
        raise


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to source video file")
    parser.add_argument("--events-dir", type=Path, default=EVENTS_DIR, help="Destination root for event folders and index files")
    parser.add_argument("--motion-only", action="store_true", help="Skip the LLM description call")
    parser.add_argument("--token-saver", action="store_true", help="Reuse the source folder's existing .json/.jpg/.keyframes instead of re-running the LLM pipeline")
    args = parser.parse_args(argv)

    if not args.video.exists():
        log.error("video not found: %s", args.video)
        return 1

    process_video(
        args.video,
        events_dir=args.events_dir,
        motion_only=args.motion_only,
        token_saver=args.token_saver,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
