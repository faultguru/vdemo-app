"""The security mode (core.security_mode): when the home mode ends."""
from __future__ import annotations

import re

from pydantic import field_validator
from pydantic_settings import BaseSettings

_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class SecuritySettings(BaseSettings):
    # Local time (HH:MM) at which the home mode turns back to normal.
    home_mode_until: str = "23:00"

    @field_validator("home_mode_until")
    @classmethod
    def _check_until(cls, v: str) -> str:
        if not _HHMM.match(v.strip()):
            raise ValueError("a local time as HH:MM, e.g. 23:00")
        return v.strip()
