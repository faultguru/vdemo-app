"""Which camera provider, which of its cameras are enabled, each camera's
detection filter (which of its clips are downloaded and processed), and which
cameras are indoors."""
from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode

from blink_watcher.core.filters import DEFAULT_FILTER, DetectionFilter

ALL_CAMERAS = {"all", "*"}  # CAMERAS value meaning every camera on the provider account


def split_list(v):
    """A setting list given as a JSON list (["A", "B"]) or comma-separated (A,B)."""
    if isinstance(v, str):
        if v.strip().startswith("["):
            return json.loads(v)
        return [c.strip() for c in v.split(",") if c.strip()]
    return v


def camera_key(name: str) -> str:
    """A camera name for comparisons: camera names match ignoring case and surrounding spaces."""
    return name.strip().lower()


class CameraSettings(BaseSettings):
    camera_provider: Literal["blink"] = "blink"

    # Enabled cameras: camera names, or "all" for every camera on the account.
    cameras: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Per camera: always | person | person_or_vehicle; cameras not listed use DEFAULT_FILTER.
    camera_filters: dict[str, DetectionFilter] = Field(default_factory=dict)
    # Cameras inside the house: in the home security mode their clips are saved without any analysis.
    indoor_cameras: Annotated[list[str], NoDecode] = Field(default_factory=list)

    poll_interval_seconds: int = 20
    # fetch --daemon, when polls fail: the wait doubles up to this; the provider reconnects
    # (reloading its session) after this many failures in a row.
    fetch_retry_max_seconds: int = 300
    fetch_reconnect_after_failures: int = 3
    # fetch --daemon, when it starts: also the days since the last stored clip (a stop or an
    # outage), at most this many days back; 0 = today only.
    fetch_catchup_days: int = 7
    # Tuning (.env.example documents each option):
    fetch_catalog_refresh_sec: float = 3600.0

    @field_validator("cameras", "indoor_cameras", mode="before")
    @classmethod
    def _split_cameras(cls, v):
        return split_list(v)

    @property
    def all_cameras(self) -> bool:
        """CAMERAS=all: every camera on the account."""
        return len(self.cameras) == 1 and camera_key(self.cameras[0]) in ALL_CAMERAS

    def camera_enabled(self, name: str) -> bool:
        """Listed in CAMERAS, or CAMERAS=all. Names match ignoring case and spaces."""
        return self.all_cameras or camera_key(name) in {camera_key(c) for c in self.cameras}

    def camera_filter(self, name: str) -> DetectionFilter:
        """The camera's detection filter: CAMERA_FILTERS, else DEFAULT_FILTER."""
        filters = {camera_key(c): f for c, f in self.camera_filters.items()}
        return filters.get(camera_key(name), DEFAULT_FILTER)

    def camera_indoor(self, name: str) -> bool:
        return camera_key(name) in {camera_key(c) for c in self.indoor_cameras}
