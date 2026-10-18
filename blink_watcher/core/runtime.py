"""Runtime folders under STORAGE_ROOT that are not storage areas: program
state, logs and downloaded model weights."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blink_watcher.settings import Settings


def _dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_dir(settings: "Settings", *parts: str) -> Path:
    """STORAGE_ROOT/state/<parts>: small files a program keeps between runs (e.g. a provider session)."""
    return _dir(settings.storage_path.joinpath("state", *parts))


def logs_dir(settings: "Settings") -> Path:
    return _dir(settings.storage_path / "logs")


def models_dir(settings: "Settings") -> Path:
    """Model weights (YOLO) and their exported variants."""
    return _dir(settings.storage_path / "models")
