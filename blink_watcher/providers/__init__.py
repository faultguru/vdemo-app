"""Camera providers behind one interface (providers.base). make_provider()
returns the one CAMERA_PROVIDER names; nothing outside this package imports a
provider's own modules or its client library (blinkpy).

Adding a provider (e.g. Ring): write providers/<name>/ with a CameraProvider
subclass (and, if it has options, a settings section), and register it in
PROVIDERS.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from blink_watcher.providers.base import (
    AuthError, Camera, CameraProvider, LiveSource, ProviderClip, ProviderError, Snapshot, TwoFactorRequired,
)

if TYPE_CHECKING:
    from blink_watcher.settings import Settings

__all__ = ["AuthError", "Camera", "CameraProvider", "LiveSource", "ProviderClip", "ProviderError", "Snapshot",
           "TwoFactorRequired", "make_provider"]

# CAMERA_PROVIDER value -> "module:class", imported on use (only the chosen provider's library is loaded).
PROVIDERS = {
    "blink": "blink_watcher.providers.blink.provider:BlinkProvider",
}


def make_provider(settings: "Settings") -> CameraProvider:
    """The CAMERA_PROVIDER's provider, not connected yet (use it as `async with`)."""
    try:
        module, cls = PROVIDERS[settings.camera_provider].split(":")
    except KeyError:
        raise ValueError(f"unknown CAMERA_PROVIDER {settings.camera_provider!r} "
                         f"(supported: {', '.join(PROVIDERS)})") from None
    return getattr(import_module(module), cls)(settings)
