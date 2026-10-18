"""The video processor: how many clips per run, and the vision checks (YOLO)."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode

from blink_watcher.settings.cameras import split_list


class ProcessingSettings(BaseSettings):
    process_limit: int = 1                      # clips per run; 0 = no limit
    # The YOLO step passes only if one of these classes moves in the motion; [] = any class.
    yolo_required_classes: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["person"])
    yolo_device: Literal["cpu", "directml"] = "cpu"

    @field_validator("yolo_required_classes", mode="before")
    @classmethod
    def _split_classes(cls, v):
        return split_list(v)
