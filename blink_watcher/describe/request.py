"""The describer's request for a frames folder written by vision.sampling
(manifest.json + JPEGs): the system prompt, and the user turn — the clip's
instructions followed by every frame as a timestamped image.

Prompts (prompts/):
  describe_video.system.txt            system prompt
  describe_video.user.txt              user-turn instructions; filled with the
                                       clip's facts, the security mode block
                                       (core.security_mode), the input-mode block
                                       and the segments block
  describe_keyframes.input_mode.txt    how the frames were sampled + YOLO classes
  describe_keyframes.crop.txt          the crop sentence, when frames are cropped
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from blink_watcher.ai import ImagePart, Part, TextPart
from blink_watcher.core.prompts import load_prompt
from blink_watcher.vision.sampling import MANIFEST

SYSTEM_PROMPT = "describe_video.system.txt"
USER_PROMPT = "describe_video.user.txt"
INPUT_MODE_PROMPT = "describe_keyframes.input_mode.txt"
CROP_PROMPT = "describe_keyframes.crop.txt"


@dataclass
class FramesRequest:
    system: str
    user_prompt: str
    parts: list[Part]   # user prompt, then a label + image per frame
    manifest: dict
    frames_sent: int


def _yolo_classes(manifest: dict) -> str:
    # Class names only: track counts aren't people counts (one person often
    # splits into several tracks), and a count would bias the people list.
    classes = {"person"} if manifest.get("person_tracks") else set()
    classes |= {tr["class"] for tr in manifest.get("object_tracks") or [] if tr.get("moved")}
    return ", ".join(sorted(classes)) or "none"


def input_mode_block(manifest: dict, n_frames: int) -> str:
    # load_prompt strips trailing whitespace: the separating space is added here.
    crop = load_prompt(CROP_PROMPT) + " " if manifest.get("roi") else ""
    return load_prompt(INPUT_MODE_PROMPT).format(n_frames=n_frames, crop=crop, yolo_classes=_yolo_classes(manifest))


def segments_block(segments: list[dict]) -> str:
    lines = [f"- Segment {s['index']}: {s['start_sec']:.2f}s → {s['end_sec']:.2f}s" for s in segments]
    return "Motion segments (use these indices for 'segment_index'):\n" + "\n".join(lines) if lines else ""


def user_prompt(info: dict, segments: list[dict], input_mode: str, segments_text: str,
                security_mode: str = "") -> str:
    return load_prompt(USER_PROMPT).format(
        security_mode_block=security_mode,
        duration_sec=info.get("duration_sec"),
        width=info.get("width"),
        height=info.get("height"),
        fps=info.get("fps"),
        num_segments=len(segments),
        input_mode_block=input_mode,
        segments_block=segments_text,
    )


def _frame_label(k: int, n: int, frame: dict) -> str:
    where = f"segment {frame['segment']}" if frame["segment"] != "gap" else "between segments"
    return f"Frame {k}/{n} @ t={frame['t']:.2f}s ({where}):"


def build_request(frames_dir: Path, security_mode: str = "") -> FramesRequest:
    manifest = json.loads((frames_dir / MANIFEST).read_text(encoding="utf-8"))
    frames = [f for f in manifest["frames"] if f.get("kind") != "crop"]  # crops: older manifests
    segments = manifest.get("segments") or []
    prompt = user_prompt(manifest["info"], segments, input_mode_block(manifest, len(frames)),
                         segments_block(segments), security_mode)
    parts: list[Part] = [TextPart(prompt)]
    for k, f in enumerate(frames, start=1):
        parts.append(TextPart(_frame_label(k, len(frames), f)))
        parts.append(ImagePart((frames_dir / f["file"]).read_bytes()))  # JPEGs sized by the sampler
    return FramesRequest(load_prompt(SYSTEM_PROMPT), prompt, parts, manifest, len(frames))
