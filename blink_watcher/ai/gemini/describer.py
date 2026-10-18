"""Reference: a whole-video describer on Gemini. Not used by the pipeline and
not wired to settings — kept as the template for adding a video-native
provider.

The live describer (blink_watcher.describe) sends sampled still frames,
because Claude takes images, not video. A model that watches video can be
given the clip itself: this one cuts the clip to its motion span (padded),
uploads it to the Gemini Files API, and asks for the same VideoAnalysis
structured output with the same prompts (prompts/describe_video.*.txt, the
input-mode block from prompts/describe_gemini.input_mode.txt and a segments
block with each segment's YOLO detections). Timestamps in the answer are
shifted back onto the original clip's timeline.

    describer = GeminiVideoDescriber(api_key, "gemini-2.5-flash")
    analysis, usage = describer.describe(video_path, segments, info)

Wiring it in would take: settings keys for the API key and model (an AI
settings section), a describer interface that both this class and
describe.VideoDescriber implement (describe(video, work folder, info,
segments) -> VideoAnalysis + call record), and a choice between them in
pipeline.process. Needs the optional dependency google-genai
(`pip install blink-watcher[gemini]`).
"""
from __future__ import annotations

import logging
import tempfile
import time
from collections import Counter
from pathlib import Path

from blink_watcher.ai.base import Usage
from blink_watcher.core.prompts import load_prompt
from blink_watcher.describe.request import SYSTEM_PROMPT, user_prompt
from blink_watcher.describe.schema import VideoAnalysis
from blink_watcher.vision.ffmpeg import trim_video
from blink_watcher.vision.motion import MotionSegment

log = logging.getLogger(__name__)

INPUT_MODE_PROMPT = "describe_gemini.input_mode.txt"
TRIM_PAD_SEC = 0.5         # keep this much around the motion span
MIN_TRIM_SAVING_SEC = 1.0  # send the whole clip when the cut would save less
UPLOAD_TIMEOUT_SEC = 120
PERSON_FPS = 5.0           # video sampling rate when a person is in the clip (hand state, bends)
DEFAULT_FPS = 1.0          # otherwise (vehicles, animals)


class GeminiVideoDescriber:
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types

        self.model = model
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def describe(self, video_path: Path, segments: list[MotionSegment], info: dict) -> tuple[VideoAnalysis, Usage]:
        """The clip's motion `segments` (with YOLO detections) and `info` come
        from vision.check."""
        if not segments:
            raise ValueError("nothing to describe: no motion segments")
        with tempfile.TemporaryDirectory(prefix="gemini-") as tmp:
            clip, offset, duration = self._cut(video_path, segments, info, Path(tmp))
            yolo = Counter(d["class"] for s in segments for d in s.detections)
            prompt = user_prompt(
                {**info, "duration_sec": round(duration, 3)},
                [{"index": s.index} for s in segments],
                load_prompt(INPUT_MODE_PROMPT).format(
                    yolo_classes=", ".join(f"{c} (×{n})" for c, n in sorted(yolo.items())) or "none"),
                self._segments_block(segments, offset),
            )
            has_person = "person" in yolo
            analysis, usage = self._ask(clip, prompt, PERSON_FPS if has_person else DEFAULT_FPS)
        if offset:
            analysis.important_frames_sec = [round(t + offset, 3) for t in analysis.important_frames_sec]
            if analysis.thumbnail_sec >= 0:
                analysis.thumbnail_sec = round(analysis.thumbnail_sec + offset, 3)
        return analysis, usage

    @staticmethod
    def _cut(video_path: Path, segments: list[MotionSegment], info: dict, tmp: Path) -> tuple[Path, float, float]:
        """(clip to send, its offset in the original, its duration): the motion
        span padded by TRIM_PAD_SEC, or the whole clip when that saves little."""
        total = float(info.get("duration_sec") or 0.0)
        first = max(0.0, min(s.start_sec for s in segments) - TRIM_PAD_SEC)
        last = min(total, max(s.end_sec for s in segments) + TRIM_PAD_SEC)
        if last <= first or total - (last - first) < MIN_TRIM_SAVING_SEC:
            return video_path, 0.0, total
        clip = tmp / f"{video_path.stem}.trim.mp4"
        trim_video(video_path, clip, first, last)
        return clip, first, last - first

    @staticmethod
    def _segments_block(segments: list[MotionSegment], offset: float) -> str:
        """Segment time ranges on the sent clip's clock, with their YOLO detections."""
        lines = []
        for s in segments:
            dets = ", ".join(f"{d['class']} ({d['confidence']:.2f})" for d in s.detections) or "none"
            lines.append(f"- Segment {s.index}: {max(0.0, s.start_sec - offset):.2f}s → "
                         f"{max(0.0, s.end_sec - offset):.2f}s (duration {s.duration_sec:.2f}s, "
                         f"peak motion {s.peak_area_ratio:.1%}). YOLOv8 detections: {dets}")
        return "Motion segments (use these indices for 'segment_index'):\n" + "\n".join(lines)

    def _upload(self, path: Path):
        """Upload to the Files API and wait until the file is ACTIVE."""
        file = self._client.files.upload(file=str(path))
        deadline = time.monotonic() + UPLOAD_TIMEOUT_SEC
        while True:
            state = getattr(file.state, "name", str(file.state))  # PROCESSING | ACTIVE | FAILED
            if state == "ACTIVE":
                return file
            if state == "FAILED":
                raise RuntimeError(f"Gemini file upload failed: {file.name}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"Gemini file upload timed out (still {state})")
            time.sleep(1.0)
            file = self._client.files.get(name=file.name)

    def _ask(self, clip: Path, prompt: str, fps: float) -> tuple[VideoAnalysis, Usage]:
        types = self._types
        file = self._upload(clip)
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[types.Part(file_data=types.FileData(file_uri=file.uri, mime_type=file.mime_type),
                                     video_metadata=types.VideoMetadata(fps=fps)),
                          prompt],
                config=types.GenerateContentConfig(system_instruction=load_prompt(SYSTEM_PROMPT),
                                                   response_mime_type="application/json",
                                                   response_schema=VideoAnalysis, temperature=0.0),
            )
        finally:
            try:
                self._client.files.delete(name=file.name)
            except Exception as e:  # the upload expires on its own
                log.warning("Gemini file %s not deleted: %s", file.name, e)
        analysis = response.parsed
        if analysis is None:
            analysis = VideoAnalysis.model_validate_json(response.text or "")
        um = response.usage_metadata
        usage = Usage(input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                      output_tokens=getattr(um, "candidates_token_count", 0) or 0,
                      cache_read_input_tokens=getattr(um, "cached_content_token_count", 0) or 0)
        return analysis, usage
