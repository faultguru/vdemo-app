"""Settings of the Blink camera provider."""
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class BlinkSettings(BaseSettings):
    blink_username: Optional[str] = None
    blink_password: Optional[str] = None
    # The PIN Blink emails when a login needs 2FA; set it only for the run that completes that login.
    blink_2fa_code: Optional[str] = None
    # Blink's clip time (created_at) is when the recording ended (uploaded), not when it started:
    # True puts a clip's recorded_at at created_at - its length, so clips of different lengths line up.
    blink_time_is_clip_end: bool = True
