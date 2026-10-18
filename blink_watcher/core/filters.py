"""Detection filters: which of a camera's clips are downloaded and processed, by
the camera provider's own object detections. Each enabled camera has one
(CAMERA_FILTERS; DEFAULT_FILTER when not set):

  always             every clip (cameras without their own person detection);
                     the video processor's motion / YOLO / LLM checks still decide
  person             clips with a person detection
  person_or_vehicle  clips with a person or a vehicle detection
"""
from __future__ import annotations

from typing import Literal, Optional, get_args

DetectionFilter = Literal["always", "person", "person_or_vehicle"]

FILTER_NAMES: tuple[str, ...] = get_args(DetectionFilter)
DEFAULT_FILTER: DetectionFilter = "person"

# filter name -> its name on the configuration page
LABELS: dict[str, str] = {"always": "Always", "person": "Person", "person_or_vehicle": "Person or vehicle"}

# filter name -> provider detections that qualify a clip (None = every clip)
WANTED: dict[str, Optional[frozenset[str]]] = {
    "always": None,
    "person": frozenset({"person"}),
    "person_or_vehicle": frozenset({"person", "vehicle"}),
}


def passes(filter_name: str, detections: list[str]) -> bool:
    wanted = WANTED[filter_name]
    return wanted is None or bool(wanted & set(detections))


def describe(filter_name: str) -> str:
    """The filter in a log line: 'always' or 'person or vehicle detection'."""
    wanted = WANTED[filter_name]
    return "always" if wanted is None else f"{' or '.join(sorted(wanted))} detection"


def summary(camera_filters: dict[str, str]) -> str:
    """CAMERA_FILTERS in a log line: 'Always: A, B; others: Person'."""
    by_filter: dict[str, list[str]] = {}
    for camera, name in camera_filters.items():
        if name != DEFAULT_FILTER:
            by_filter.setdefault(name, []).append(camera.strip())
    parts = [f"{LABELS[name]}: {', '.join(sorted(cams))}" for name, cams in sorted(by_filter.items())]
    return "; ".join(parts + [f"{'others' if parts else 'every camera'}: {LABELS[DEFAULT_FILTER]}"])
