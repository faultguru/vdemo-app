"""Send a frame_sampler `.details` folder to Claude and save the analysis.

Uses the same system prompt and user-prompt template as the Gemini video
describer (prompts/describe_video.*.txt). Only the input-mode block differs:
Claude gets timestamped still frames instead of a video, so the block explains
the sampling and tells the model to judge transfers from before/after state
across frames rather than requiring the exact moment to be pictured.

Two backends:
  cli (default) — runs `claude -p` headless with the frames as image blocks,
                  the describer prompt as the system prompt, no tools, and the
                  VideoAnalysis JSON schema. Uses your Claude Code login.
  api           — Anthropic SDK with the OAuth token from
                  ~/.claude/.credentials.json (same as ClaudeDescriber).

Files written to the details dir:
  claude.request.jsonl + claude.request.args.json  (cli)  — the exact request;
      replay by hand with `claude <args...> < claude.request.jsonl`
  claude.request.api.json                          (api)  — the Messages API body
  claude.result.json — the parsed VideoAnalysis plus model, user prompt, usage,
      stop reason and timing.

Usage:
    python -m blink_watcher.describe_keyframes videos/claude-test/clip.details
    python -m blink_watcher.describe_keyframes videos/claude-test/clip.details --backend api
    python -m blink_watcher.describe_keyframes videos/claude-test/clip.details --dry-run
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from blink_watcher.describers import ClaudeDescriber, VideoAnalysis, load_prompt

log = logging.getLogger("blink_watcher.describe_keyframes")

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 16000  # thinking is on by default on current models; leave room for it plus the JSON
CLI_TIMEOUT_SEC = 900

# Request files written next to the frames (see write_request).
REQUEST_JSONL = "claude.request.jsonl"      # stdin for `claude -p` (stream-json user message with the images)
REQUEST_ARGS = "claude.request.args.json"   # argv for `claude -p` (system prompt + schema inline)
REQUEST_API = "claude.request.api.json"     # Messages API body (--backend api)
RESULT_FILE = "claude.result.json"          # response: analysis + model, usage, timing


def _yolo_summary(manifest: dict) -> str:
    # Class names only — track counts aren't people counts (one person often
    # splits into several tracks), and a count would bias the people list.
    classes = {"person"} if manifest.get("person_tracks") else set()
    classes |= {tr["class"] for tr in manifest.get("object_tracks") or [] if tr.get("moved")}
    return ", ".join(sorted(classes)) or "none"


def _input_mode_block(manifest: dict, n_frames: int) -> str:
    crop = (
        "Every frame is cropped to the same region around the activity; nothing outside it moves. "
        if manifest.get("roi") else ""
    )
    return (
        f"You are given {n_frames} still frames from this clip, in time order, each labelled with its "
        f"timestamp in seconds from the start of the clip. {crop}"
        "Frames are spaced about one second apart across the whole span of motion, with denser runs "
        "(about four per second) around moments when a person bends, crouches, appears or leaves. "
        "A short action can fall between two frames, so judge what happened from the state BEFORE and "
        "AFTER across frames: compare what a person holds in an earlier frame with what they hold in a "
        "later frame, and what is on the ground or a surface before and after. A gap between frames is "
        "not by itself uncertainty — rule (e) of the OBJECT TRANSFER RULE applies only when the frames "
        "before AND after leave the hand state unclear.\n\n"
        f"YOLOv8 ground-truth detections across the whole clip: {_yolo_summary(manifest)}. "
        "YOLO detects moving subjects (people, vehicles, animals) only — it cannot detect carried items, "
        "so packages, boxes, bags and food containers never appear in that list. Use the YOLO class names "
        "for each event's 'object' field (rule 3c), and report carried or placed items through "
        "'people[].objects', the action enums and the descriptions."
    )


def _segments_block(manifest: dict) -> str:
    lines = [
        f"- Segment {s['index']}: {s['start_sec']:.2f}s → {s['end_sec']:.2f}s"
        for s in manifest.get("segments") or []
    ]
    return "Motion segments (use these indices for 'segment_index'):\n" + "\n".join(lines) if lines else ""


def _segment_label(seg: str) -> str:
    return f"segment {seg}" if seg != "gap" else "between segments"


def build_request(keyframes_dir: Path) -> tuple[str, list[dict], dict]:
    """Return (user_prompt_text, content_blocks, manifest)."""
    manifest = json.loads((keyframes_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = [f for f in manifest["frames"] if f.get("kind") != "crop"]
    info = manifest["info"]

    user_prompt = load_prompt("describe_video.user.txt").format(
        duration_sec=info.get("duration_sec"),
        width=info.get("width"),
        height=info.get("height"),
        fps=info.get("fps"),
        num_segments=len(manifest.get("segments") or []),
        num_described=len(manifest.get("segments") or []),
        input_mode_block=_input_mode_block(manifest, len(frames)),
        segments_block=_segments_block(manifest),
    )

    content: list[dict] = [{"type": "text", "text": user_prompt}]
    for k, f in enumerate(frames, start=1):
        content.append({
            "type": "text",
            "text": f"Frame {k}/{len(frames)} @ t={f['t']:.2f}s ({_segment_label(f['segment'])}):",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode((keyframes_dir / f["file"]).read_bytes()).decode("ascii"),
            },
        })
    return user_prompt, content, manifest


# ---------- Request files ----------

def _cli_args(model: str, system_prompt: str) -> list[str]:
    """argv for `claude -p` (minus the executable): the describer prompt as the
    whole system prompt, no tools, no MCP, no project settings, and the
    VideoAnalysis schema for structured output."""
    return [
        "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json", "--verbose",
        "--model", model,
        "--system-prompt", system_prompt,
        "--json-schema", json.dumps(VideoAnalysis.model_json_schema()),
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--no-session-persistence",
    ]


def write_request(out_dir: Path, model: str, backend: str, system_prompt: str, content: list[dict]) -> list[Path]:
    """Write the exact request to files so it can be inspected or replayed.

    cli: REQUEST_JSONL is the stdin for `claude` and REQUEST_ARGS its argv —
         replay with `claude <args...> < claude.request.jsonl`.
    api: REQUEST_API is the Messages API body (structured output as
         output_config.format, the wire form of messages.parse).
    """
    if backend == "cli":
        jsonl = out_dir / REQUEST_JSONL
        jsonl.write_text(json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n",
                         encoding="utf-8")
        args = out_dir / REQUEST_ARGS
        args.write_text(json.dumps(_cli_args(model, system_prompt), indent=2), encoding="utf-8")
        return [jsonl, args]
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": content}],
        "output_config": {"format": {"type": "json_schema", "schema": VideoAnalysis.model_json_schema()}},
    }
    api = out_dir / REQUEST_API
    api.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return [api]


# ---------- Backends ----------

def _call_api(model: str, system_prompt: str, content: list[dict]) -> dict:
    client = ClaudeDescriber(model)._client
    result = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        output_format=VideoAnalysis,
    )
    u = result.usage
    parsed = result.parsed_output
    return {
        "analysis": parsed,
        "stop_reason": result.stop_reason,
        "usage": {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        },
        "request_id": getattr(result, "_request_id", None),
        "raw_text": None if parsed is not None else "".join(
            getattr(b, "text", "") for b in result.content if getattr(b, "type", "") == "text"
        ),
    }


def _call_cli(request_dir: Path) -> dict:
    """Run `claude -p` from the request files written by write_request()."""
    exe = shutil.which("claude")
    if not exe:
        raise SystemExit("`claude` CLI not found on PATH")
    args = json.loads((request_dir / REQUEST_ARGS).read_text(encoding="utf-8"))
    # Run from an empty temp dir so no project CLAUDE.md / memory is picked up.
    with tempfile.TemporaryDirectory(prefix="describe-keyframes-") as cwd, \
            open(request_dir / REQUEST_JSONL, encoding="utf-8") as stdin:
        proc = subprocess.run(
            [exe, *args], stdin=stdin, capture_output=True,
            text=True, encoding="utf-8", cwd=cwd, timeout=CLI_TIMEOUT_SEC,
        )

    result = None
    for line in proc.stdout.splitlines():
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "result":
            result = evt
    if result is None:
        raise RuntimeError(
            f"claude -p exited {proc.returncode} without a result event:\n"
            f"stderr: {proc.stderr[-2000:]}\nstdout: {proc.stdout[-2000:]}"
        )

    structured = result.get("structured_output")
    parsed = None
    if structured is not None:
        parsed = VideoAnalysis.model_validate(structured)
    elif result.get("result"):
        try:
            parsed = VideoAnalysis.model_validate_json(result["result"])
        except ValueError:
            parsed = None
    return {
        "analysis": parsed,
        "stop_reason": result.get("stop_reason") or result.get("subtype"),
        "usage": result.get("usage"),
        "request_id": result.get("session_id"),
        "cost_usd_equivalent": result.get("total_cost_usd"),
        "is_error": result.get("is_error"),
        "raw_text": None if parsed is not None else result.get("result"),
    }


# ---------- Entry points ----------

def analyze_keyframes(keyframes_dir: Path, model: str, backend: str = "cli") -> tuple[Optional[VideoAnalysis], dict]:
    """Build + write the request for a .keyframes folder, call Claude, and
    return (analysis or None, metadata). Used by the CLI below and by
    describers.ClaudeCliDescriber in the live pipeline."""
    user_prompt, content, manifest = build_request(keyframes_dir)
    n_images = sum(1 for c in content if c["type"] == "image")
    system_prompt = load_prompt("describe_video.system.txt")
    files = write_request(keyframes_dir, model, backend, system_prompt, content)
    log.info("%s: %d frame(s), ~%s image tokens, model=%s, backend=%s, request=%s",
             keyframes_dir.name, n_images, (manifest.get("totals") or {}).get("approx_image_tokens"),
             model, backend, ", ".join(f.name for f in files))

    t0 = time.perf_counter()
    res = _call_cli(keyframes_dir) if backend == "cli" else _call_api(model, system_prompt, content)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    log.info("stop_reason=%s in %d ms", res["stop_reason"], elapsed_ms)

    parsed = res.pop("analysis")
    meta = {
        "model": model,
        "backend": backend,
        "frames_sent": n_images,
        "duration_ms": elapsed_ms,
        "request_files": [f.name for f in files],
        "trim": manifest.get("trim"),  # frame timestamps are on the trimmed timeline when set
        **res,
        "user_prompt": user_prompt,
    }
    return parsed, meta


def describe(keyframes_dir: Path, model: str, backend: str = "cli", dry_run: bool = False) -> Optional[Path]:
    if dry_run:
        user_prompt, content, _ = build_request(keyframes_dir)
        files = write_request(keyframes_dir, model, backend, load_prompt("describe_video.system.txt"), content)
        log.info("dry run: wrote %s", ", ".join(str(f) for f in files))
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
        print(user_prompt)
        return None

    parsed, meta = analyze_keyframes(keyframes_dir, model, backend)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "keyframes_dir": keyframes_dir.as_posix(),
        **meta,
        "analysis": parsed.model_dump() if parsed is not None else None,
    }
    path = keyframes_dir / RESULT_FILE
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path)
    if parsed is None:
        log.error("no parsed output (stop_reason=%s) — see raw_text in %s", meta["stop_reason"], path)
    return path


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keyframes_dir", type=Path, help="A <clip>.details folder written by frame_sampler")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")
    parser.add_argument("--backend", choices=("cli", "api"), default="cli",
                        help="cli = `claude -p` with your Claude Code login (default); api = Anthropic SDK")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the request files and print the user prompt; do not call the model")
    args = parser.parse_args(argv)

    if not (args.keyframes_dir / "manifest.json").exists():
        log.error("no manifest.json in %s", args.keyframes_dir)
        return 1
    describe(args.keyframes_dir, args.model, backend=args.backend, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
