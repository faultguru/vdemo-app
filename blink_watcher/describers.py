"""Video-describer providers (Claude and Gemini).

Abstraction over the LLM that turns OpenCV motion segments + YOLO detections
into structured event JSON. Both providers emit the same ``VideoAnalysis``
shape so the pipeline in ``describe_video.py`` is provider-agnostic.

Selection is via the ``VIDEO_DESCRIBER`` env var (``claude`` or ``gemini``).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from blink_watcher.describe_video import MotionSegment

log = logging.getLogger("blink_watcher.describers")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
KEYFRAME_MAX_SIDE = 1600
MAX_SEGMENTS_TO_DESCRIBE = 12


# ---------- Shared structured-output schema ----------

ActionType = Literal[
    "package_dropoff",
    "package_pickup",
    "package_carrying",
    "person_walking",
    "person_running",
    "person_standing",
    "person_talking",
    "person_crouching",
    "person_knocking",
    "person_checking_phone",
    "door_opening",
    "gate_opening",
    "mailbox_checking",
    "vehicle_arriving",
    "vehicle_departing",
    "vehicle_passing",
    "person_entering_vehicle",
    "person_exiting_vehicle",
    "dog_walking",
    "animal_present",
    "noise",
]

PersonType = Literal[
    "kid",
    "teen",
    "adult",
    "senior",
    "delivery_person",
    "mail_carrier",
    "police",
    "service_worker",
    "visitor",
    "unknown",
]

ObjectType = Literal[
    "package",
    "phone",
    "bag",
    "box",
    "clipboard",
    "envelope",
    "umbrella",
    "tool",
    "food_container",
    "dog",
    "cat",
    "bicycle",
    "stroller",
]


class Event(BaseModel):
    segment_index: int = Field(description="Which motion segment this event belongs to.")
    object: str = Field(description="Short canonical noun for the moving object: 'person', 'car', 'dog', 'bicycle', 'package', 'animal', 'noise', etc.")
    details: str = Field(description="Distinguishing attributes of this specific object: color, size, clothing, orientation. Empty string if nothing distinctive.")
    description: str = Field(description="One sentence describing what this object is literally doing.")


class Person(BaseModel):
    appearance: str
    appearance_details: str = ""
    type: PersonType = "unknown"
    estimated_age: Optional[int] = None
    actions: list[ActionType] = Field(default_factory=list)
    objects: list[ObjectType] = Field(default_factory=list)


class Vehicle(BaseModel):
    appearance: str
    appearance_details: str = ""
    actions: list[ActionType] = Field(default_factory=list)


class Summary(BaseModel):
    description: str
    actions: list[ActionType] = Field(default_factory=list)
    people: list[Person] = Field(default_factory=list)
    vehicles: list[Vehicle] = Field(default_factory=list)
    ignore: bool = False


def _empty_summary(description: str = "", ignore: bool = False) -> Summary:
    return Summary(
        description=description,
        actions=[],
        people=[],
        vehicles=[],
        ignore=ignore,
    )


class VideoAnalysis(BaseModel):
    summary: Summary = Field(description="Top-level summary of the whole clip.")
    events: list[Event] = Field(
        default_factory=list,
        description="One entry per distinct moving object per segment.",
    )
    important_frames_sec: list[float] = Field(
        default_factory=list,
        description=(
            "Timestamps (seconds from clip start) of the single most informative "
            "moments in the clip — one per distinct action state you want a viewer "
            "to be able to glance at. Typically 2–6 values. Include the thumbnail "
            "moment in this list too. Empty list if there is no real motion."
        ),
    )
    thumbnail_sec: float = Field(
        default=-1.0,
        description=(
            "Single best timestamp (seconds from clip start) to use as the video "
            "thumbnail — the one frame that captures what the clip is about. "
            "Should be one of the values in important_frames_sec. Use -1.0 if "
            "there is no real motion."
        ),
    )


# ---------- Prompt loading ----------

def load_prompt(name: str) -> str:
    """Read a prompt file, stripping '#' comment lines and trailing whitespace."""
    path = PROMPTS_DIR / name
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not ln.lstrip().startswith("#")]
    return "\n".join(kept).strip()


# ---------- Base interface ----------

class VideoDescriber(ABC):
    provider: str = ""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def describe(
        self,
        video_path: Path,
        segments: "list[MotionSegment]",
        frames_bgr: dict[int, np.ndarray],
        info: dict,
    ) -> tuple[VideoAnalysis, dict]:
        """Return (analysis, usage_dict). Usage dict shape is provider-defined
        but always includes integer token counts where available."""


# ---------- JPEG helpers (shared with Claude path) ----------

def _jpeg_bytes(frame_bgr: np.ndarray, max_side: int = KEYFRAME_MAX_SIDE) -> bytes:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _jpeg_b64(frame_bgr: np.ndarray) -> str:
    return base64.standard_b64encode(_jpeg_bytes(frame_bgr)).decode("ascii")


# ---------- Claude implementation ----------

class ClaudeDescriber(VideoDescriber):
    provider = "claude"

    def __init__(self, model: str):
        super().__init__(model)
        import anthropic  # local import so Gemini-only runs don't require it
        self._anthropic = anthropic
        self._client = self._make_client()

    def _load_oauth_token(self) -> str:
        path = Path.home() / ".claude" / ".credentials.json"
        if not path.exists():
            raise SystemExit(
                f"Claude credentials not found at {path}. "
                "Run `claude` once to sign in, or set up OAuth credentials."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth") or {}
        token = oauth.get("accessToken")
        if not token:
            raise SystemExit(f"No accessToken in {path}")
        return token

    def _make_client(self):
        token = self._load_oauth_token()
        os.environ.pop("ANTHROPIC_API_KEY", None)
        return self._anthropic.Anthropic(
            auth_token=token,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )

    def describe(
        self,
        video_path: Path,
        segments: "list[MotionSegment]",
        frames_bgr: dict[int, np.ndarray],
        info: dict,
    ) -> tuple[VideoAnalysis, dict]:
        if not segments:
            return VideoAnalysis(summary=_empty_summary("No motion detected.", ignore=True), events=[]), {}

        to_describe = segments[:MAX_SEGMENTS_TO_DESCRIBE]
        yolo_counts: dict[str, int] = {}
        for seg in to_describe:
            for d in seg.detections:
                cls = d.get("class")
                if cls:
                    yolo_counts[cls] = yolo_counts.get(cls, 0) + 1
        yolo_classes_summary = (
            ", ".join(f"{cls} (×{n})" for cls, n in sorted(yolo_counts.items()))
            if yolo_counts else "none"
        )
        claude_mode_block = (
            f"Keyframes for the first {len(to_describe)} segment(s) are attached as images, "
            "sampled evenly across each segment. Compare keyframes within a segment pixel-by-pixel "
            "to decide whether a subject is actually changing position, pose, or presence. "
            f"YOLOv8 ground-truth detections across the whole clip: {yolo_classes_summary}. "
            "These are the ONLY object types confirmed to be present — see rule (3c) below."
        )
        user_prompt = load_prompt("describe_video.user.txt").format(
            duration_sec=info.get("duration_sec"),
            width=info.get("width"),
            height=info.get("height"),
            fps=info.get("fps"),
            num_segments=len(segments),
            num_described=len(to_describe),
            input_mode_block=claude_mode_block,
            segments_block="",
        )
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        fps = info.get("fps") or 25.0

        for seg in to_describe:
            detections_line = ""
            if seg.detections:
                det_str = ", ".join(
                    f"{d['class']} ({d['confidence']:.2f})" for d in seg.detections
                )
                detections_line = (
                    f"\nGROUND TRUTH objects (from YOLOv8 on the keyframes): {det_str}. "
                    f"Use ONLY these class names for the 'object' field. Do NOT invent other "
                    f"object types. If YOLO found 'car', say 'car' — not 'truck', 'pickup', 'SUV', etc."
                )
            content.append({
                "type": "text",
                "text": (
                    f"\n--- Segment {seg.index} — {seg.start_sec:.2f}s to {seg.end_sec:.2f}s "
                    f"(duration {seg.duration_sec:.2f}s, peak motion {seg.peak_area_ratio:.1%}) ---"
                    f"{detections_line}"
                ),
            })
            bx, by, bw, bh = seg.bbox
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            for k, t_sec in enumerate(seg.keyframe_times_sec, start=1):
                frame_idx = int(round(t_sec * fps))
                frame = frames_bgr.get(frame_idx)
                if frame is None:
                    continue
                fh, fw = frame.shape[:2]
                cx0 = max(0, bx - pad_x)
                cy0 = max(0, by - pad_y)
                cx1 = min(fw, bx + bw + pad_x)
                cy1 = min(fh, by + bh + pad_y)
                crop = frame[cy0:cy1, cx0:cx1] if (cx1 > cx0 and cy1 > cy0) else frame
                content.append({
                    "type": "text",
                    "text": f"Keyframe {k}/{len(seg.keyframe_times_sec)} @ t={t_sec:.2f}s:",
                })
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _jpeg_b64(crop),
                    },
                })

        content[-1]["cache_control"] = {"type": "ephemeral"}

        log.info("asking Claude (%s) to describe %d segment(s)", self.model, len(to_describe))
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "system": [{
                "type": "text",
                "text": load_prompt("describe_video.system.txt"),
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [{"role": "user", "content": content}],
            "output_format": VideoAnalysis,
        }
        if "opus-4-6" in self.model or "sonnet-4-6" in self.model:
            kwargs["thinking"] = {"type": "adaptive"}
        result = self._client.messages.parse(**kwargs)
        u = getattr(result, "usage", None)
        usage = {
            "input_tokens": getattr(u, "input_tokens", 0),
            "output_tokens": getattr(u, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        log.info(
            "claude usage: input=%d output=%d total=%d",
            usage["input_tokens"], usage["output_tokens"], usage["total_tokens"],
        )
        return result.parsed_output, usage


# ---------- Gemini implementation ----------

class GeminiDescriber(VideoDescriber):
    provider = "gemini"

    def __init__(self, model: str):
        super().__init__(model)
        from google import genai  # local import
        from google.genai import types as genai_types
        self._genai = genai
        self._types = genai_types
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=api_key)

    def _upload_video(self, video_path: Path):
        """Upload the video to Gemini Files API and wait until it is ACTIVE."""
        log.info("uploading %s to Gemini Files API", video_path.name)
        video_file = self._client.files.upload(file=str(video_path))
        # Poll for ACTIVE. State comes back as an enum whose .name is PROCESSING|ACTIVE|FAILED.
        deadline = time.monotonic() + 120
        while True:
            state = getattr(video_file.state, "name", str(video_file.state))
            if state == "ACTIVE":
                return video_file
            if state == "FAILED":
                raise RuntimeError(f"Gemini file upload failed: {video_file.name}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"Gemini file upload timed out (still {state})")
            time.sleep(1.0)
            video_file = self._client.files.get(name=video_file.name)

    def _segments_block(self, segments: "list[MotionSegment]", offset_sec: float = 0.0) -> str:
        """Render segment ranges. Times are shifted by -offset_sec so they are
        expressed in the trimmed-clip timeline Gemini actually sees."""
        lines = []
        for seg in segments:
            dets = ", ".join(
                f"{d['class']} ({d['confidence']:.2f})" for d in seg.detections
            ) or "none"
            s = max(0.0, seg.start_sec - offset_sec)
            e = max(0.0, seg.end_sec - offset_sec)
            lines.append(
                f"- Segment {seg.index}: {s:.2f}s → {e:.2f}s "
                f"(duration {seg.duration_sec:.2f}s, peak motion {seg.peak_area_ratio:.1%}). "
                f"YOLOv8 detections: {dets}"
            )
        return "\n".join(lines)

    def _trim_to_motion(
        self,
        video_path: Path,
        segments: "list[MotionSegment]",
        info: dict,
        pad_sec: float = 0.5,
    ) -> tuple[Path, float, float]:
        """Write a temp .mp4 containing only [first_motion - pad, last_motion + pad].

        Returns (trimmed_path, offset_sec, trimmed_duration_sec).
        Falls back to the original path when trimming saves almost nothing
        (< 1s) so we don't pay re-encode cost for no gain.
        """
        total = float(info.get("duration_sec") or 0.0)
        first = max(0.0, min(s.start_sec for s in segments) - pad_sec)
        last = min(total, max(s.end_sec for s in segments) + pad_sec)
        if last <= first or (total - (last - first)) < 1.0:
            log.info("trim would save <1s; sending full video")
            return video_path, 0.0, total

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or info.get("fps") or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame = int(round(first * fps))
        end_frame = int(round(last * fps))

        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"{video_path.stem}.", suffix=".trim.mp4"
        )
        os.close(tmp_fd)
        out_path = Path(tmp_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        written = 0
        for _ in range(end_frame - start_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        cap.release()
        writer.release()

        trimmed_dur = written / fps if fps else (last - first)
        log.info(
            "trimmed %s → %s (%.2fs → %.2fs, saved %.0f%%)",
            video_path.name, out_path.name, total, trimmed_dur,
            (1 - trimmed_dur / total) * 100 if total else 0,
        )
        return out_path, first, trimmed_dur

    def describe(
        self,
        video_path: Path,
        segments: "list[MotionSegment]",
        frames_bgr: dict[int, np.ndarray],
        info: dict,
    ) -> tuple[VideoAnalysis, dict]:
        if not segments:
            return VideoAnalysis(summary=_empty_summary("No motion detected.", ignore=True), events=[]), {}

        to_describe = segments[:MAX_SEGMENTS_TO_DESCRIBE]

        # Trim to [first_motion_start - pad, last_motion_end + pad] so Gemini
        # only bills tokens on the motion span, not the whole clip.
        trimmed_path, offset_sec, trimmed_dur = self._trim_to_motion(
            video_path, to_describe, info,
        )

        yolo_counts: dict[str, int] = {}
        for seg in to_describe:
            for d in seg.detections:
                cls = d.get("class")
                if cls:
                    yolo_counts[cls] = yolo_counts.get(cls, 0) + 1
        yolo_classes_summary = (
            ", ".join(f"{cls} (×{n})" for cls, n in sorted(yolo_counts.items()))
            if yolo_counts else "none"
        )
        gemini_mode_block = (
            "The full video is attached — watch ONLY the listed time ranges below. "
            f"YOLOv8 ground-truth detections across the whole clip: {yolo_classes_summary}. "
            "These are the ONLY object types confirmed to be present — see rule (3c) below."
        )

        user_prompt = load_prompt("describe_video.user.txt").format(
            duration_sec=round(trimmed_dur, 3),
            width=info.get("width"),
            height=info.get("height"),
            fps=info.get("fps"),
            num_segments=len(segments),
            num_described=len(to_describe),
            input_mode_block=gemini_mode_block,
            segments_block=self._segments_block(to_describe, offset_sec=offset_sec),
        )
        system_prompt = load_prompt("describe_video.system.txt")

        video_file = None
        try:
            video_file = self._upload_video(trimmed_path)

            # People need finer temporal resolution to read behavioral detail
            # (placing vs dropping a package, crouching, holding a phone…).
            # Vehicles and animals are fine at the default 1 FPS.
            has_person = any(
                any(d.get("class") == "person" for d in seg.detections)
                for seg in to_describe
            )
            sample_fps = 5.0 if has_person else 1.0
            log.info("gemini video sampling at %.0f fps (person detected: %s)", sample_fps, has_person)

            video_part = self._types.Part(
                file_data=self._types.FileData(
                    file_uri=video_file.uri,
                    mime_type=video_file.mime_type,
                ),
                video_metadata=self._types.VideoMetadata(fps=sample_fps),
            )

            log.info("asking Gemini (%s) to describe %d segment(s)", self.model, len(to_describe))
            config = self._types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=VideoAnalysis,
                temperature=0.0,
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=[video_part, user_prompt],
                config=config,
            )

            parsed = getattr(response, "parsed", None)
            if parsed is None:
                raw = getattr(response, "text", "") or ""
                try:
                    parsed = VideoAnalysis.model_validate_json(raw)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"Gemini response was not valid VideoAnalysis JSON: {raw[:500]}") from exc

            # Gemini sees the trimmed clip — shift its timestamps back to the
            # original video's timeline so the caller can seek the raw file.
            if offset_sec:
                parsed.important_frames_sec = [
                    round(t + offset_sec, 3) for t in parsed.important_frames_sec
                ]
                if parsed.thumbnail_sec >= 0:
                    parsed.thumbnail_sec = round(parsed.thumbnail_sec + offset_sec, 3)

            um = getattr(response, "usage_metadata", None)
            usage = {
                "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "cached_content_token_count": getattr(um, "cached_content_token_count", 0) or 0,
            }
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
            log.info(
                "gemini usage: input=%d output=%d total=%d",
                usage["input_tokens"], usage["output_tokens"], usage["total_tokens"],
            )

            return parsed, usage
        finally:
            if video_file is not None:
                try:
                    self._client.files.delete(name=video_file.name)
                except Exception:  # noqa: BLE001
                    pass
            if trimmed_path != video_path:
                try:
                    trimmed_path.unlink()
                except OSError:
                    pass


# ---------- Claude via `claude -p` ----------

class ClaudeCliDescriber(VideoDescriber):
    """Claude through the headless Claude Code CLI (`claude -p`, your Claude
    Code login) on frames picked by frame_sampler from a trimmed copy of the
    clip — ~1 fps over the whole motion span plus dense bursts around
    bends/entries/exits, all cropped to the activity region — instead of
    ClaudeDescriber's 8 keyframes per segment.
    Prompt and request files: see describe_keyframes."""

    provider = "claude-cli"

    def describe(
        self,
        video_path: Path,
        segments: "list[MotionSegment]",
        frames_bgr: dict[int, np.ndarray],
        info: dict,
    ) -> tuple[VideoAnalysis, dict]:
        if not segments:
            return VideoAnalysis(summary=_empty_summary("No motion detected.", ignore=True), events=[]), {}

        # Local imports: both modules import this one.
        from blink_watcher.describe_keyframes import analyze_keyframes
        from blink_watcher.frame_sampler import sample_frames

        with tempfile.TemporaryDirectory(prefix=f"{video_path.stem}.frames.") as tmp:
            frames_dir = Path(tmp)
            sample_frames(video_path, frames_dir, info=info, segments=segments)
            parsed, meta = analyze_keyframes(frames_dir, self.model, backend="cli")
        if parsed is None:
            raise RuntimeError(
                f"claude -p returned no structured output (stop_reason={meta.get('stop_reason')}): "
                f"{(meta.get('raw_text') or '')[:500]}"
            )
        # Claude saw the trimmed video's timeline — shift back so the caller
        # can seek the original file.
        offset = (meta.get("trim") or {}).get("offset_sec") or 0.0
        if offset:
            parsed.important_frames_sec = [round(t + offset, 3) for t in parsed.important_frames_sec]
            if parsed.thumbnail_sec >= 0:
                parsed.thumbnail_sec = round(parsed.thumbnail_sec + offset, 3)
        return parsed, meta.get("usage") or {}


# ---------- Factory ----------

def make_describer() -> VideoDescriber:
    provider = os.getenv("VIDEO_DESCRIBER", "claude").strip().lower()
    if provider == "claude":
        return ClaudeDescriber(os.getenv("CLAUDE_MODEL", "claude-haiku-4-5"))
    if provider == "claude-cli":
        return ClaudeCliDescriber(os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"))
    if provider == "gemini":
        model = os.getenv("GEMINI_VIDEO_MODEL")
        if not model:
            raise SystemExit("GEMINI_VIDEO_MODEL must be set in .env")
        return GeminiDescriber(model)
    raise SystemExit(f"Unknown VIDEO_DESCRIBER={provider!r} (expected 'claude', 'claude-cli' or 'gemini')")
