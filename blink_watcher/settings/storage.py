"""Where everything is stored, and the local time zone."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings

from blink_watcher.core.paths import PROJECT_ROOT

StorageType = Literal["localdisk"]


class StorageSettings(BaseSettings):
    # Every runtime file lives under STORAGE_ROOT: one folder per storage area
    # (<AREA>_PREFIX, type <AREA>_STORAGE_TYPE) plus state/, logs/ and models/.
    storage_root: Path = Path("./data-storage")
    raw_videos_storage_type: StorageType = "localdisk"
    raw_videos_prefix: str = "videos.raw"              # original clips + .metadata.json, as downloaded
    processed_videos_storage_type: StorageType = "localdisk"
    processed_videos_prefix: str = "videos.processed"  # inbox/ = clips waiting to be processed
    ignored_videos_storage_type: StorageType = "localdisk"
    ignored_videos_prefix: str = "videos.ignored"      # .metadata.json of skipped / unimportant clips (no video)
    temp_videos_storage_type: StorageType = "localdisk"
    temp_videos_prefix: str = "videos.temp"            # per-clip working files (frames, LLM request/response)
    live_snapshots_storage_type: StorageType = "localdisk"
    live_snapshots_prefix: str = "videos.live"         # live monitor snapshots
    chat_storage_type: StorageType = "localdisk"
    chat_prefix: str = "chat"                          # memory, events-inbox/, outbox/, <date>/ day files

    timezone: str = "America/Los_Angeles"

    @property
    def storage_path(self) -> Path:
        """STORAGE_ROOT as an absolute path (a relative one is taken from the project folder)."""
        root = Path(self.storage_root)
        return root if root.is_absolute() else (PROJECT_ROOT / root).resolve()
