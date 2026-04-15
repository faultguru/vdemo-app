import json
import logging
import os
from pathlib import Path

from aiohttp import ClientSession
from blinkpy.auth import Auth, BlinkTwoFARequiredError, LoginError
from blinkpy.blinkpy import Blink

from .config import Settings

log = logging.getLogger(__name__)


class AuthError(RuntimeError):
    pass


class TwoFARequired(AuthError):
    pass


async def connect(settings: Settings) -> Blink:
    log.info("auth: creating aiohttp session")
    session = ClientSession()
    blink = Blink(session=session)

    creds = {"username": settings.blink_username, "password": settings.blink_password}
    if settings.session_file.exists():
        log.info("auth: loading cached session from %s", settings.session_file)
        try:
            creds.update(json.loads(settings.session_file.read_text()))
        except json.JSONDecodeError:
            log.warning("auth: session file unreadable, falling back to password login")
    else:
        log.info("auth: no cached session, will do fresh login for %s", settings.blink_username)

    blink.auth = Auth(creds, no_prompt=True, session=session)

    try:
        log.info("auth: calling blink.start() ...")
        started = await blink.start()
        if started is False or not blink.auth.token:
            await session.close()
            raise AuthError(
                "auth: Blink rejected the login (no token returned). "
                "Most likely cause: wrong username/password in .env. "
                "Confirm you can log in to the Blink mobile app with these credentials."
            )
        log.info("auth: blink.start() returned OK, token acquired")
    except BlinkTwoFARequiredError:
        log.warning("auth: 2FA required — Blink has emailed a verification code")
        code = os.environ.get("BLINK_2FA_CODE")
        if not code:
            await session.close()
            raise TwoFARequired(
                "Blink requires 2FA. Check your email for the PIN, then re-run with "
                "BLINK_2FA_CODE=<pin> set in the environment."
            )
        log.info("auth: submitting 2FA code from BLINK_2FA_CODE env var")
        ok = await blink.auth.complete_2fa_login(code)
        if not ok:
            await session.close()
            raise AuthError("auth: 2FA verification failed (wrong or expired code)")
        log.info("auth: 2FA verification accepted")
        # finish the OAuth v2 flow that start() was doing
        log.info("auth: finalizing blink startup after 2FA")
        await blink.start()
        log.info("auth: blink.start() returned OK after 2FA")
    except AuthError:
        raise
    except LoginError as e:
        await session.close()
        raise AuthError(f"auth: login rejected by Blink (bad credentials?): {e}") from e
    except Exception as e:
        await session.close()
        raise AuthError(f"auth: unexpected error during login: {type(e).__name__}: {e}") from e

    log.info(
        "auth: connected — networks=%s cameras=%s",
        list((blink.networks or {}).keys()),
        list((blink.cameras or {}).keys()),
    )
    _persist_session(blink, settings.session_file)
    log.info("auth: session persisted to %s", settings.session_file)
    return blink


def _persist_session(blink: Blink, path: Path) -> None:
    data = {
        "token": blink.auth.token,
        "host": blink.auth.host,
        "region_id": blink.auth.region_id,
        "client_id": blink.auth.client_id,
        "account_id": blink.auth.account_id,
        "user_id": getattr(blink.auth, "user_id", None),
        "unique_id": getattr(blink.auth, "unique_id", None),
        "device_id": getattr(blink.auth, "device_id", None),
        "refresh_token": getattr(blink.auth, "refresh_token", None),
        "hardware_id": getattr(blink.auth, "hardware_id", None),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
