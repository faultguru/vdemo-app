"""Event stitching: related clips grouped into activities by fixed rules (no AI)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class StitchSettings(BaseSettings):
    stitch_enabled: bool = True
    stitch_overlap_slack_sec: float = 3.0         # overlap: other camera, same moment
    stitch_continuation_gap_sec: float = 60.0     # continuation: same camera, next clip
    stitch_transition_gap_sec: float = 45.0       # transition: other camera, shortly after
    stitch_track_gap_sec: float = 600.0           # track: same subject per the event's track field
    stitch_min_score_overlap: float = 0.35
    stitch_min_score_continuation: float = 0.25
    stitch_min_score_transition: float = 0.5
    # {"Camera A": ["Camera B"], ...}, either direction.
    stitch_camera_neighbors: dict[str, list[str]] = Field(default_factory=dict)
    stitch_neighbor_bonus: float = 0.1
    stitch_non_neighbor_penalty: float = 0.15
    stitch_max_span_sec: float = 900.0            # longest activity; longer ones are cut at their largest gap
    stitch_default_clip_sec: float = 30.0
    stitch_vocab_file: Optional[Path] = None      # word lists; None = the built-in stitch/vocab.json
