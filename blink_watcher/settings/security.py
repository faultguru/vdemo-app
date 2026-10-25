"""The security mode (core.security_mode): when the home mode ends."""
from __future__ import annotations

import re

from pydantic import field_validator
from pydantic_settings import BaseSettings

_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class SecuritySettings(BaseSettings):
    # Local time (HH:MM) at which the home mode turns back to normal.
    home_mode_until: str = "23:00"
    # Auto-away: when the notifier sees a household vehicle (as long-term memory describes it)
    # drive off while the mode is HOME, the mode becomes AWAY; when it comes back while the
    # mode is AWAY, HOME again (pipeline.notify.auto_away).
    auto_away: bool = False
    # Vehicle-only clips of the same camera (not downloaded: the filter wants a person) that
    # start within this many seconds of an event are listed to the notifier: the car pulling
    # in or driving off is often such a clip.
    auto_away_clip_window_sec: int = 180

    @field_validator("home_mode_until")
    @classmethod
    def _check_until(cls, v: str) -> str:
        if not _HHMM.match(v.strip()):
            raise ValueError("a local time as HH:MM, e.g. 23:00")
        return v.strip()
