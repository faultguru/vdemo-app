"""The camera-provider interface: what the app needs from a camera cloud (Blink, ...).

A provider is an async context manager holding one session with the
provider's cloud; everything else in the app goes through it:

  list_cameras()                     the account's cameras (Camera)
  list_clips(cameras, since, until)  recorded clips in a time range (ProviderClip)
  download_clip(clip)                a clip's .mp4, streamed
  snapshots(cameras)                 a new picture of each camera (Snapshot)
  live_stream(camera)                the camera's live view, as an ffmpeg input (LiveSource)

Cameras are addressed by name; names match ignoring case and surrounding
spaces (settings.cameras.camera_key), as in CAMERAS.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import AsyncIterator, Optional

from blink_watcher.settings.cameras import camera_key

# ProviderClip.trigger values every provider maps its own onto (others pass through unchanged).
MOTION, SNAPSHOT, LIVE_VIEW = "motion", "snapshot", "live_view"


class ProviderError(RuntimeError):
    """A provider operation failed; the message is meant for the log / the user."""


class AuthError(ProviderError):
    """The provider rejected the login, or credentials are missing."""


class TwoFactorRequired(AuthError):
    """The login needs a verification code the user has to supply."""


@dataclass(frozen=True)
class Camera:
    name: str
    provider: str                                     # "blink"
    id: str                                           # the provider's camera id
    model: Optional[str] = None                       # the provider's camera type / product name
    has_own_person_detection: Optional[bool] = None   # None: not known from the provider

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(name=d["name"], provider=d["provider"], id=str(d["id"]), model=d.get("model"),
                   has_own_person_detection=d.get("has_own_person_detection"))


@dataclass(frozen=True)
class ProviderClip:
    provider: str
    provider_clip_id: str
    camera: str
    recorded_at: datetime              # UTC
    duration_sec: Optional[float]
    trigger: Optional[str]             # MOTION | SNAPSHOT | LIVE_VIEW | the provider's own value
    detections: list[str]              # the provider's own object labels ("person", "vehicle"); [] if none
    raw: dict = field(default_factory=dict, compare=False, hash=False)   # the provider's clip record
    provider_time: Optional[datetime] = None   # the provider's own clip time (UTC), when it differs from recorded_at


@dataclass(frozen=True)
class Snapshot:
    """One camera's answer to snapshots(): the picture (JPEG), or why there is none."""
    camera: str
    data: Optional[bytes] = None
    error: Optional[str] = None


class LiveSource(ABC):
    """A camera's live view as ffmpeg input. `input_args` are ffmpeg's input
    options; with `uses_stdin` the stream is written to ffmpeg's stdin by
    start(). stop() ends the stream on the provider's side."""
    kind: str                       # for logs, e.g. "rtsps"
    input_args: list[str]
    uses_stdin: bool = False

    async def start(self, stdin: Optional[asyncio.StreamWriter]) -> None:
        """Begin feeding the stream (ffmpeg is already running)."""

    async def stop(self) -> None:
        """End the stream."""


class UrlLiveSource(LiveSource):
    """A live view ffmpeg reads from a URL itself (RTSP, HLS, ...)."""

    def __init__(self, url: str, input_options: Optional[list[str]] = None):
        self.kind = url.split(":", 1)[0]
        self.input_args = [*(input_options or []), "-i", url]


class CameraProvider(ABC):
    name: str   # the provider's CAMERA_PROVIDER value, e.g. "blink"

    async def __aenter__(self) -> "CameraProvider":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    @abstractmethod
    async def connect(self) -> None:
        """Open the session (raises AuthError / TwoFactorRequired / ProviderError)."""

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def list_cameras(self) -> list[Camera]: ...

    @abstractmethod
    def list_clips(self, cameras: Optional[list[str]], since: Optional[datetime] = None,
                   until: Optional[datetime] = None) -> AsyncIterator[ProviderClip]:
        """Clips of `cameras` (None = every camera) recorded in [since, until), in any order."""

    @abstractmethod
    def download_clip(self, clip: ProviderClip) -> AsyncIterator[bytes]:
        """The clip's .mp4 in chunks (for Storage.write_stream)."""

    @abstractmethod
    async def snapshots(self, cameras: list[str]) -> dict[str, Snapshot]:
        """A new picture of each camera, by name. Batched: the provider may
        take a while per camera (Blink spaces camera commands ~2 s apart)."""

    @abstractmethod
    async def live_stream(self, camera: str) -> LiveSource:
        """Start the camera's live view (raises ProviderError if it does not start)."""


def find_camera(cameras: dict[str, object], name: str):
    """`cameras` (name -> provider object) entry for `name`, matched like CAMERAS names; None if absent."""
    if name in cameras:
        return cameras[name]
    key = camera_key(name)
    return next((c for n, c in cameras.items() if camera_key(n) == key), None)
