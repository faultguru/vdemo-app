"""Blink login with a cached session.

The session (tokens, host, ids, expiry) is kept in
STORAGE_ROOT/state/blink/session.json: loaded on connect, written after the
login and whenever blinkpy refreshes the token. Null values are never written
nor loaded — a null (e.g. device_id) would override blinkpy's default and end
up as a None request header, which breaks every token refresh.

Without a usable session, BLINK_USERNAME / BLINK_PASSWORD log in. If Blink then
asks for 2FA it emails a PIN: re-run with BLINK_2FA_CODE=<pin> set.

Blink changes its login API without notice and blinkpy follows; an older
blinkpy than MIN_BLINKPY is known to break (its token refresh is rejected with
a 406 once the first token expires), so connect() logs an error for it.
"""
from __future__ import annotations

import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from aiohttp import ClientSession
from blinkpy.auth import Auth, BlinkTwoFARequiredError, LoginError
from blinkpy.blinkpy import Blink

from blink_watcher.core.runtime import state_dir
from blink_watcher.providers.base import AuthError, TwoFactorRequired
from blink_watcher.settings import Settings

log = logging.getLogger(__name__)

MIN_BLINKPY = (0, 25, 9)  # keep in step with pyproject.toml

# Fields of blinkpy's Auth kept in the session file.
SESSION_FIELDS = ("token", "host", "region_id", "client_id", "account_id", "user_id", "unique_id",
                  "refresh_token", "expires_in", "expiration_date", "hardware_id")


def session_file(settings: Settings) -> Path:
    return state_dir(settings, "blink") / "session.json"


def _load_session(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        cached = json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("Blink session file unreadable, logging in with the password: %s", path)
        return {}
    log.info("Blink session loaded from %s", path)
    return {k: v for k, v in cached.items() if v is not None}


def _persist_session(blink: Blink, path: Path) -> None:
    data = {name: getattr(blink.auth, name, None) for name in SESSION_FIELDS}
    data["device_id"] = blink.auth.data.get("device_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in data.items() if v is not None}, indent=2))


async def connect(settings: Settings) -> Blink:
    """A started Blink session (networks and cameras loaded). Raises AuthError /
    TwoFactorRequired; the aiohttp session is closed on failure."""
    _check_blinkpy()
    path = session_file(settings)
    cached = _load_session(path)
    if not cached and not (settings.blink_username and settings.blink_password):
        raise AuthError(f"no Blink session ({path}) and BLINK_USERNAME / BLINK_PASSWORD are not set")
    creds = {"username": settings.blink_username, "password": settings.blink_password, **cached}

    session = ClientSession()
    blink = Blink(session=session)
    blink.auth = Auth(creds, no_prompt=True, session=session, callback=lambda: _persist_session(blink, path))
    try:
        await _start(blink, settings)
    except BaseException:
        await session.close()
        raise
    log.info("Connected to Blink: networks %s, cameras %s",
             list((blink.networks or {}).keys()), list((blink.cameras or {}).keys()))
    _persist_session(blink, path)
    return blink


def _check_blinkpy() -> None:
    try:
        installed = version("blinkpy")
        parts = tuple(int(p) for p in installed.split(".")[:3] if p.isdigit())
    except (PackageNotFoundError, ValueError):
        return
    if parts < MIN_BLINKPY:
        log.error("blinkpy %s is too old for Blink's current login (need %s or newer): pip install -U blinkpy",
                  installed, ".".join(map(str, MIN_BLINKPY)))


async def _start(blink: Blink, settings: Settings) -> None:
    try:
        started = await blink.start()
        if started is False or not blink.auth.token:
            raise AuthError("Blink rejected the login (no token returned); check BLINK_USERNAME / BLINK_PASSWORD "
                            "by logging in to the Blink app with them")
    except BlinkTwoFARequiredError:
        log.warning("Blink requires 2FA: a verification PIN was emailed")
        if not settings.blink_2fa_code:
            raise TwoFactorRequired("Blink requires 2FA: re-run with BLINK_2FA_CODE=<PIN from the email> set")
        if not await blink.auth.complete_2fa_login(settings.blink_2fa_code):
            raise AuthError("Blink 2FA verification failed (wrong or expired code)")
        log.info("Blink 2FA verification accepted")
        await blink.start()  # finish the OAuth flow start() was doing
    except AuthError:
        raise
    except LoginError as e:
        raise AuthError(f"Blink login rejected (bad credentials?): {e}") from e
    except Exception as e:
        raise AuthError(f"Blink login failed: {type(e).__name__}: {e}") from e
