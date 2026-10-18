"""Storage abstraction for everything the app persists (raw clips, metadata, ...).

Storage is split into *areas*, one per kind of data (RAW_VIDEOS, PROCESSED_VIDEOS, CHAT, ...),
each configured on its own — type and folder — so e.g. raw videos could move
to S3 while the chat area stays on local disk. Only "localdisk" exists today.

Callers get an area with make_storage(settings, AREA) and address files in it
by *key* — a forward-slash path relative to the area, e.g.
"2026-10-17/Front_Door/2026-10-17-00-03-18.mp4" — and never build filesystem
paths themselves, so another backend can be added behind the same interface.

Config per area: <AREA>_STORAGE_TYPE and <AREA>_PREFIX (the folder name); a
localdisk area lives at STORAGE_ROOT/<prefix>. Key layout helpers are at the
bottom.
"""
from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, AsyncIterable, Iterator
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from blink_watcher.settings import Settings


class Storage(ABC):
    kind: str
    folder: str = ""   # the area's folder name — the first folder under STORAGE_ROOT, e.g. "videos.raw"

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None:
        """Atomic: readers see either the old object or the complete new one."""

    @abstractmethod
    async def write_stream(self, key: str, chunks: AsyncIterable[bytes]) -> int:
        """Atomic like write_bytes, for large objects; returns bytes written."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """No-op if the key does not exist."""

    @abstractmethod
    def size(self, key: str) -> int: ...

    @abstractmethod
    def list(self, prefix: str = "") -> Iterator[str]:
        """Keys under `prefix`, recursively."""

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """A local file path for tools that need one (OpenCV, ffmpeg). On
        localdisk this is the stored file itself — treat it as read-only."""

    @abstractmethod
    def uri(self, key: str) -> str:
        """Human-readable location, for logs."""

    def read_text(self, key: str) -> str:
        return self.read_bytes(key).decode("utf-8")

    def write_text(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode("utf-8"))

    def read_json(self, key: str):
        return json.loads(self.read_text(key))

    def write_json(self, key: str, obj) -> None:
        self.write_text(key, json.dumps(obj, indent=2, ensure_ascii=False))

    def delete_tree(self, prefix: str) -> None:
        """Delete every key under `prefix`. No-op if there are none."""
        for key in list(self.list(prefix)):
            self.delete(key)

    def total_size(self, prefix: str = "") -> int:
        return sum(self.size(k) for k in self.list(prefix))


class LocalDiskStorage(Storage):
    kind = "localdisk"
    _PART = ".part"

    def __init__(self, root: Path, folder: str = ""):
        self.root = Path(root)
        self.folder = folder or self.root.name

    def _path(self, key: str) -> Path:
        rel = PurePosixPath(key)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"storage key must be relative and stay under the root: {key!r}")
        return self.root.joinpath(*rel.parts)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + self._PART)
        part.write_bytes(data)
        part.replace(path)

    async def write_stream(self, key: str, chunks: AsyncIterable[bytes]) -> int:
        import aiofiles
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + self._PART)
        written = 0
        try:
            async with aiofiles.open(part, "wb") as f:
                async for chunk in chunks:
                    await f.write(chunk)
                    written += len(chunk)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(path)
        return written

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def delete_tree(self, prefix: str) -> None:
        if not prefix:
            raise ValueError("refusing to delete the whole storage area")
        shutil.rmtree(self._path(prefix), ignore_errors=True)

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def list(self, prefix: str = "") -> Iterator[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.is_dir():
            return
        for dirpath, _, files in os.walk(base):
            for name in files:
                if name.endswith(self._PART):
                    continue
                yield Path(dirpath, name).relative_to(self.root).as_posix()

    def local_path(self, key: str) -> Path:
        return self._path(key)

    def uri(self, key: str) -> str:
        return str(self._path(key))


# Storage areas.
RAW_VIDEOS = "raw_videos"               # original clips + .metadata.json, as downloaded
PROCESSED_VIDEOS = "processed_videos"   # processing state; inbox/ = clips waiting to be processed
IGNORED_VIDEOS = "ignored_videos"       # .metadata.json of skipped / unimportant clips, no video
TEMP_VIDEOS = "temp_videos"             # per-clip working folders: <date>/<camera>/<stamp>/ (event/ = the notifier)
LIVE_SNAPSHOTS = "live_snapshots"       # live monitor snapshots: <date>/<camera>/<stamp>.jpg
CHAT = "chat"                           # memory/memory.md, events-inbox/, outbox/ + <date>/ day notes, events index, chat history


def area_config(settings: "Settings", area: str) -> tuple[str, str]:
    """(storage type, prefix) for an area, from <AREA>_STORAGE_TYPE / <AREA>_PREFIX."""
    try:
        return getattr(settings, f"{area}_storage_type"), getattr(settings, f"{area}_prefix")
    except AttributeError:
        raise ValueError(f"unknown storage area {area!r}") from None


def make_storage(settings: "Settings", area: str) -> Storage:
    kind, prefix = area_config(settings, area)
    if kind == "localdisk":
        return LocalDiskStorage(settings.storage_path / prefix, folder=prefix)
    raise ValueError(f"unsupported {area.upper()}_STORAGE_TYPE {kind!r} (supported: localdisk)")


# ---------- Key layout ----------

def raw_clip_key(settings: "Settings", camera: str, ts: datetime) -> str:
    """Key in the RAW_VIDEOS area: <YYYY-MM-DD>/<Camera_Name>/<YYYY-MM-DD-HH-MM-SS>.mp4, local time."""
    local_ts = ts.astimezone(ZoneInfo(settings.timezone))
    return f"{local_ts:%Y-%m-%d}/{camera_folder(camera)}/{local_ts:%Y-%m-%d-%H-%M-%S}.mp4"


def camera_folder(camera: str) -> str:
    """Camera name as a path segment: " Front Door" -> "Front_Door"."""
    return camera.strip().replace("/", "_").replace(" ", "_")


def clip_name(camera: str, file_name: str) -> str:
    """<stamp>-<CAMERA><suffix> for a "<stamp><suffix>" file name (or a bare
    stamp), e.g. ("Front Door", "2026-10-17-00-03-18.metadata.json") ->
    2026-10-17-00-03-18-FRONT_DOOR.metadata.json. Timestamp first, so names
    sort oldest to newest across cameras; the camera keeps same-second clips
    of different cameras apart. Used for inbox files, event ids, outbox messages."""
    stamp, dot, suffix = file_name.partition(".")
    return f"{stamp}-{camera_folder(camera).upper()}{dot}{suffix}"


def inbox_key(camera: str, file_name: str) -> str:
    """Key in the PROCESSED_VIDEOS area: inbox/<stamp>-<CAMERA>.metadata.json
    (clip_name), e.g. inbox/2026-10-17-00-03-18-FRONT_DOOR.metadata.json."""
    return f"inbox/{clip_name(camera, file_name)}"


def file_link(area_folder: str, key: str) -> dict:
    """{"folder", "video_file"} of a file in a storage area, as stored in .metadata.json
    ("original" / "processed"); folder is relative to STORAGE_ROOT."""
    return {
        "folder": str(PurePosixPath(area_folder) / PurePosixPath(key).parent),
        "video_file": PurePosixPath(key).name,
    }


def link_key(storage: Storage, link: dict) -> str:
    """Inverse of file_link: the file's key inside `storage` (whose folder must match)."""
    parts = PurePosixPath(link["folder"]).parts
    if not parts or parts[0] != storage.folder:
        raise ValueError(f"folder {link['folder']!r} is not in the {storage.folder!r} area")
    return str(PurePosixPath(*parts[1:], link["video_file"]))


def metadata_key(video_key: str) -> str:
    """The clip's .metadata.json, next to it."""
    p = PurePosixPath(video_key)
    return str(p.with_name(f"{p.stem}.metadata.json"))
