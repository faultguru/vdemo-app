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
    # Service health (core.health, supervisor watchdog): Telegram is told once a service
    # has been failing this long (0 = never), and again when it works again; the
    # supervisor restarts fetch when it has not polled for HEALTH_STALE_MINUTES (0 = never)
    # and tells Telegram when a service exits CRASH_ALERT_AFTER_RESTARTS times in a row
    # without staying up.
    health_alert_after_minutes: int = 15
    health_stale_minutes: int = 15
    crash_alert_after_restarts: int = 5
    # Errors the services log are also sent to Telegram (core.alerts), the same kind
    # of error at most once per ALERT_REPEAT_MINUTES.
    alert_on_errors: bool = True
    alert_repeat_minutes: int = 360
    # Telegram is told when the storage disk has less free space than this (0 = never).
    disk_min_free_gb: float = 5.0
    # Tuning (.env.example documents each option):
    control_start_wait_sec: int = 30
    control_stop_wait_sec: int = 60
    inbox_poll_sec: float = 1.0
    inbox_retry_sec: float = 60.0
    inbox_retry_max_sec: float = 3600.0
    inbox_alert_after_failures: int = 5
    inbox_unreadable_alert_sec: float = 120.0
    alert_max_per_hour: int = 6
    restart_min_sec: float = 2.0
    restart_max_sec: float = 60.0
    restart_stable_sec: float = 60.0
    stop_timeout_sec: float = 20.0
    settings_settle_sec: float = 2.0
    watchdog_sec: float = 60.0
    asleep_detect_sec: float = 30.0

    @property
    def storage_path(self) -> Path:
        """STORAGE_ROOT as an absolute path (a relative one is taken from the project folder)."""
        root = Path(self.storage_root)
        return root if root.is_absolute() else (PROJECT_ROOT / root).resolve()
