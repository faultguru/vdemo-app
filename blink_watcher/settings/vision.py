"""Video analysis tuning (the video processor's quick check, motion check, YOLO,
tracking, trimming, frames for the AI, thumbnails). The defaults are tuned on
real clips; see docs/video_analysis.md before changing them."""
from __future__ import annotations

from pydantic_settings import BaseSettings


class VisionSettings(BaseSettings):
    triage_samples: int = 6
    triage_max_side: int = 160
    triage_pixel_delta: int = 20
    triage_changed_ratio: float = 0.005
    motion_warmup_frames: int = 35
    motion_learning_rate: float = 0.0008
    motion_min_contour_area_ratio: float = 0.0015
    motion_min_frame_area_ratio: float = 0.0025
    motion_max_frame_area_ratio: float = 0.9
    motion_flash_mean_delta: float = 12.0
    motion_min_segment_sec: float = 0.4
    motion_merge_gap_sec: float = 1.0
    motion_keyframes_per_segment: int = 8
    motion_max_side: int = 960
    motion_frame_stride: int = 3
    yolo_conf_threshold: float = 0.2
    yolo_motion_overlap: float = 0.2
    yolo_min_shift_ratio: float = 0.03
    track_scan_fps: float = 5.0
    track_detect_conf: float = 0.35
    track_yolo_batch: int = 16
    track_iou: float = 0.3
    track_max_gap_sec: float = 1.0
    track_min_samples: int = 3
    track_person_move_ratio: float = 0.05
    track_object_move_ratio: float = 0.03
    trim_pad_sec: float = 1.0
    trim_crf: int = 23
    trim_max_rate_ratio: float = 0.9
    sample_span_pad_sec: float = 0.5
    sample_baseline_fps: float = 1.0
    sample_burst_fps: float = 4.0
    sample_burst_half_sec: float = 0.75
    sample_baseline_near_burst_sec: float = 0.15
    sample_max_frames: int = 48
    sample_frame_max_side: int = 960
    sample_roi_pad_ratio: float = 0.2
    sample_roi_min_pad_ratio: float = 0.08
    sample_roi_min_saving: float = 0.1
    bend_standing_percentile: int = 80
    bend_aspect_ratio: float = 0.7
    bend_top_margin_ratio: float = 0.02
    bend_edge_guard_sec: float = 1.5
    bend_merge_gap_sec: float = 0.4
    sample_trigger_merge_sec: float = 1.0
    thumbnail_max_side: int = 1600
    thumbnail_quality: int = 92
