"""VideoDescriber: a clip -> VideoAnalysis through an LLM that takes images.

The clip's frames are sampled (vision.sampling: ~1 fps across the activity
plus dense bursts around bends, entries and exits, cropped to the activity
region), sent with the describer prompts (describe.request) to the
configured LLM provider (blink_watcher.ai, model DESCRIBER_MODEL, effort
DESCRIBER_EFFORT), and the structured answer is parsed as VideoAnalysis.

Files in the frames folder: the JPEGs + manifest.json, the LLM request files
(REQUEST_FILES_GLOB) and RESULT_FILE — the answer on the sampled clip's
timeline plus the user prompt, frames sent, the trim, model, usage, cost and
timing.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from blink_watcher.ai import LLMClient, LLMRequest, LLMResponse, make_llm
from blink_watcher.ai.base import Progress
from blink_watcher.describe import RESULT_FILE
from blink_watcher.describe.request import build_request
from blink_watcher.describe.schema import VideoAnalysis
from blink_watcher.vision.focus import Focus
from blink_watcher.vision.motion import MotionSegment
from blink_watcher.vision.sampling import MANIFEST, sample_frames

if TYPE_CHECKING:
    from blink_watcher.settings import Settings

log = logging.getLogger(__name__)


@dataclass
class Description:
    analysis: Optional[VideoAnalysis]  # None: no valid structured answer (see response.raw_text)
    response: LLMResponse
    frames_sent: int
    trim: Optional[dict]               # the sampler's trim; analysis times are on the trimmed clip
    user_prompt: str

    def result(self, frames_dir: Path) -> dict:
        """The RESULT_FILE body."""
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "frames_dir": frames_dir.as_posix(),
            "frames_sent": self.frames_sent,
            "trim": self.trim,
            **self.response.record(),
            "user_prompt": self.user_prompt,
            "analysis": self.analysis.model_dump() if self.analysis is not None else None,
        }


def _quiet(msg: str, level: int = logging.INFO) -> None:
    log.log(level, msg)


class VideoDescriber:
    """`progress(message, level)` gets one-line status notices (frames sampled,
    request sent, response received, retries)."""

    def __init__(self, llm: LLMClient, model: str, effort: Optional[str] = None,
                 progress: Optional[Progress] = None):
        self.llm = llm
        self.model = model
        self.effort = effort or None
        self.progress = progress or _quiet

    @classmethod
    def from_settings(cls, settings: "Settings", progress: Optional[Progress] = None) -> "VideoDescriber":
        return cls(make_llm(settings), settings.describer_model, settings.describer_effort, progress)

    def describe_frames(self, frames_dir: Path, security_mode: str = "") -> Description:
        """Ask the LLM about a sampled frames folder and write RESULT_FILE there.
        `security_mode`: the security mode block (core.security_mode.prompt_block) for
        the prompt. Times in the answer are on the sampled clip's timeline."""
        req = build_request(frames_dir, security_mode)
        self.progress(f"llm             Sending {req.frames_sent} frames to {self.model}...")
        t0 = time.perf_counter()
        response = self.llm.ask(LLMRequest(req.system, req.parts, VideoAnalysis, self.model, self.effort),
                                frames_dir, self.progress)
        usage = response.usage
        self.progress(
            f"llm     {time.perf_counter() - t0:5.1f}s  Response received ({response.stop_reason}): "
            f"{usage.total_input_tokens} input tokens, {usage.output_tokens} output tokens"
            + (f", ~${response.cost_usd:.3f}" if response.cost_usd else ""))
        description = Description(response.parsed, response, req.frames_sent, req.manifest.get("trim"),
                                  req.user_prompt)
        path = frames_dir / RESULT_FILE
        path.write_text(json.dumps(description.result(frames_dir), indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Wrote %s", path)
        return description

    def describe(self, video_path: Path, frames_dir: Path, info: dict, segments: list[MotionSegment], *,
                 trim: bool = True, focus: Optional[Focus] = None, reuse_frames: bool = False,
                 security_mode: str = "") -> Description:
        """Sample the clip's frames into `frames_dir` and describe them; the
        answer's times are shifted back onto `video_path`'s timeline.
        `info` + `segments`: the clip's motion pass; `focus`: its YOLO scan
        (skips a second one); `trim=False` when the clip is already trimmed;
        `reuse_frames`: send the frames an earlier run left in `frames_dir`;
        `security_mode`: the security mode block for the prompt.
        Raises RuntimeError when the LLM gives no valid answer."""
        t0 = time.perf_counter()
        manifest_path = frames_dir / MANIFEST
        reused = reuse_frames and manifest_path.is_file()
        if not reused:
            manifest_path = sample_frames(video_path, frames_dir, info=info, segments=segments, trim=trim, focus=focus)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        totals, roi = manifest.get("totals") or {}, manifest.get("roi")
        self.progress(
            f"frames  {time.perf_counter() - t0:5.1f}s  "
            f"{totals.get('frames', 0)} frames {'reused from previous run' if reused else 'extracted'}"
            + (f", cropped to activity region ({roi['area_ratio']:.0%} of frame)" if roi else ", full frame")
            + f", ~{totals.get('approx_image_tokens', 0)} image tokens")

        description = self.describe_frames(frames_dir, security_mode)
        analysis = description.analysis
        if analysis is None:
            raise RuntimeError(f"{self.llm.name} returned no structured output "
                               f"(stop_reason={description.response.stop_reason}): "
                               f"{(description.response.raw_text or '')[:500]}")
        offset = (description.trim or {}).get("offset_sec") or 0.0
        if offset:
            analysis.important_frames_sec = [round(t + offset, 3) for t in analysis.important_frames_sec]
            if analysis.thumbnail_sec >= 0:
                analysis.thumbnail_sec = round(analysis.thumbnail_sec + offset, 3)
        return description
