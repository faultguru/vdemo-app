"""The video processor's run context: the storage areas, the run options, and
one queued clip with the keys derived from it."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Callable, Optional

from blink_watcher.chat.store import EVENTS_INBOX
from blink_watcher.core.filters import DEFAULT_FILTER
from blink_watcher.core.security_mode import ModeState
from blink_watcher.core.storage import (
    CHAT, IGNORED_VIDEOS, PROCESSED_VIDEOS, RAW_VIDEOS, TEMP_VIDEOS, Storage, clip_name, make_storage, metadata_key,
)

if TYPE_CHECKING:
    from blink_watcher.settings import Settings

INBOX = "inbox"


@dataclass
class Areas:
    """The storage areas the processor reads and writes."""
    raw: Storage        # original clips (read only)
    processed: Storage  # inbox/ + kept clips and their metadata
    temp: Storage       # per-clip work folders
    ignored: Storage    # metadata of ignored clips
    chat: Storage       # events-inbox/ of new events

    @classmethod
    def from_settings(cls, settings: "Settings") -> "Areas":
        return cls(*(make_storage(settings, a)
                     for a in (RAW_VIDEOS, PROCESSED_VIDEOS, TEMP_VIDEOS, IGNORED_VIDEOS, CHAT)))


@dataclass
class Options:
    limit: int                   # PROCESS_LIMIT
    preserve: bool
    redo: bool
    required_classes: list[str] = field(default_factory=list)  # YOLO_REQUIRED_CLASSES
    daemon: bool = False         # keep polling the inbox
    filter_override: Optional[str] = None  # --filter: one detection filter for every camera
    camera_filter: Callable[[str], str] = lambda camera: DEFAULT_FILTER  # CAMERA_FILTERS

    security_mode: Callable[[], ModeState] = ModeState  # core.security_mode.current, read per clip
    indoor_camera: Callable[[str], bool] = lambda camera: False  # INDOOR_CAMERAS

    def filter_of(self, camera: str) -> str:
        return self.filter_override or self.camera_filter(camera)


@dataclass
class Clip:
    """One inbox entry and the keys derived from it."""
    inbox_key: str  # inbox/<stamp>-<CAMERA>.metadata.json in the processed area
    md: dict        # the queued .metadata.json
    key: str        # <YYYY-MM-DD>/<Camera_Name>/<stamp>.mp4 — same in the raw, processed and ignored areas

    @property
    def stamp(self) -> str:
        return PurePosixPath(self.key).stem

    @property
    def camera(self) -> str:
        return self.md["camera"]

    def sibling(self, suffix: str) -> str:
        """<YYYY-MM-DD>/<Camera_Name>/<stamp><suffix>, next to the video."""
        return str(PurePosixPath(self.key).with_name(self.stamp + suffix))

    @property
    def metadata_key(self) -> str:
        return metadata_key(self.key)

    @property
    def thumbnail_key(self) -> str:
        return self.sibling(".jpg")

    @property
    def event_key(self) -> str:
        return self.sibling(".event.json")

    @property
    def event_id(self) -> str:
        """<stamp>-<CAMERA>: unique across cameras, like the inbox file names."""
        return clip_name(self.camera, self.stamp)

    @property
    def event_inbox_key(self) -> str:
        """events-inbox/<stamp>-<CAMERA>.event.json in the chat area."""
        return f"{EVENTS_INBOX}/{self.event_id}.event.json"

    @property
    def work_key(self) -> str:
        """<YYYY-MM-DD>/<Camera_Name>/<stamp> — the clip's work folder in the temp
        area: cut video, metadata, frames, LLM request / result."""
        return self.sibling("")

    def work_file(self, suffix: str) -> str:
        """<work folder>/<stamp><suffix>, e.g. ".mp4", ".metadata.json"."""
        return f"{self.work_key}/{self.stamp}{suffix}"


def transfer(src: Storage, src_key: str, dst: Storage, dst_key: str, keep_source: bool) -> None:
    """Move a file between storage areas — or copy it, with keep_source."""
    dst.write_bytes(dst_key, src.read_bytes(src_key))
    if not keep_source:
        src.delete(src_key)
