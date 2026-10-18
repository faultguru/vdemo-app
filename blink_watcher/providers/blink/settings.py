"""Settings of the Blink camera provider."""
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class BlinkSettings(BaseSettings):
    blink_username: Optional[str] = None
    blink_password: Optional[str] = None
    # The PIN Blink emails when a login needs 2FA; set it only for the run that completes that login.
    blink_2fa_code: Optional[str] = None
