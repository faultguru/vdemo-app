"""The Telegram chat bot service.

Long-polls Telegram for incoming messages and hands them to the ChatAgent
(the same agent the notifier uses for incoming events). Only messages from
the authorized chat (TELEGRAM_CHAT_ID) are processed; everything else is
dropped. "/clear" empties today's chat history window; "/normal", "/home",
"/away" and "/vacation" set the security mode (core.security_mode), "/mode" shows it.

Also sends the messages the notifier queues in chat/outbox/: each file is one
message — text + the event's thumbnail as the photo — which is delivered to
Telegram, then recorded as sent (chat history + "sent_at" in the event list)
and removed. A message Telegram refuses (a bad request) is sent again
without its photo, else moved to outbox/failed/ (an error: Telegram is told
with a short message). When Telegram cannot be reached, the outbox and the
incoming poll each log one warning and retry with a growing wait (up to
RETRY_MAX_SECONDS), and log a line once it works again. Ctrl+C stops.

Health (core.health): the chat-bot reports each pass of its incoming poll,
and Telegram's state as the telegram dependency (failing while it cannot be
reached or refuses the bot).
Logs go to stdout and STORAGE_ROOT/logs/chat-bot.log.

Usage:
    python -m blink_watcher chat-bot
    python -m blink_watcher chat-bot -v     # also the other modules' INFO logs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from contextlib import suppress
from pathlib import Path
from typing import Optional

import aiohttp

from blink_watcher.chat import store as chat_store
from blink_watcher.chat.agent import ChatAgent
from blink_watcher.chat.telegram import TelegramClient, TelegramError
from blink_watcher.chat.tools import EventTools
from blink_watcher.core import security_mode
from blink_watcher.core.health import Dependency, Health
from blink_watcher.core.logs import setup_logging
from blink_watcher.settings import Settings, load_settings
from blink_watcher.settings import get_settings

_settings = get_settings()

log = logging.getLogger("blink_watcher.chat.bot")

PROGRAM = "chat-bot"
OUTBOX_POLL_SECONDS = _settings.telegram_outbox_poll_sec
RETRY_SECONDS = _settings.telegram_retry_sec              # first wait after Telegram could not be reached; doubles while it goes on
RETRY_MAX_SECONDS = _settings.telegram_retry_max_sec
FAILED = "failed"        # outbox/failed/: messages Telegram refused
TRANSIENT = (aiohttp.ClientError, asyncio.TimeoutError, OSError)


class Outage:
    """Telegram not reachable: one warning when it starts, a growing wait, one line when it ends."""

    def __init__(self, what: str, health: Optional[Dependency] = None):
        self.what = what
        self.health = health  # the telegram dependency
        self.since: Optional[float] = None
        self.wait = RETRY_SECONDS

    def failed(self, e: BaseException, note: str = "") -> float:
        """The seconds to wait before trying again."""
        retry_after = getattr(e, "retry_after", None)
        if self.health:
            self.health.failed(f"not reachable: {_describe(e)}")
        if self.since is None:
            self.since, self.wait = time.monotonic(), RETRY_SECONDS
            log.warning(f"{self.what}: Telegram not reachable ({_describe(e)}){note} - retrying, "
                        f"less often while it goes on")
        else:
            self.wait = min(self.wait * 2, RETRY_MAX_SECONDS)
        return max(self.wait, retry_after or 0)

    def ok(self) -> None:
        if self.health:
            self.health.ok()
        if self.since is not None:
            log.info(f"{self.what}: Telegram reachable again after {(time.monotonic() - self.since) / 60:.1f} min")
            self.since = None


def _describe(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
TYPING_EVERY_SECONDS = 4  # Telegram shows "typing..." for about 5 s


# ---------- outbox ----------

async def send_outbox_message(path: Path, telegram: TelegramClient, settings: Settings) -> None:
    """Send one chat/outbox message (text + photo) and record it as sent
    (chat.store.record_sent). When Telegram cannot be reached this raises and
    the file stays for the next try; a message Telegram refuses is sent
    without its photo, else moved to outbox/failed/."""
    try:
        message = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("outbox: bad file %s (%s); deleting", path.name, e)
        path.unlink(missing_ok=True)
        return
    text = (message.get("text") or "").strip()
    photo = settings.storage_path / message["photo"] if message.get("photo") else None
    if photo is not None and not photo.exists():
        log.warning("outbox: %s photo not found: %s - sending text only", path.name, photo)
        photo = None
    try:
        try:
            if photo is not None:
                await telegram.send_photo(photo, caption=text or None)
            elif text:
                await telegram.send_message(text)
        except TelegramError as e:
            if not (e.permanent and photo is not None and text):
                raise
            log.warning(f"outbox: {path.name} refused with its photo ({e}) - sending the text only")
            photo = None
            await telegram.send_message(text)
    except TelegramError as e:
        if not e.permanent:
            raise
        failed = path.parent / FAILED / path.name
        failed.parent.mkdir(exist_ok=True)
        path.replace(failed)
        log.error(f"outbox: Telegram refused {path.name} ({e}) - moved to outbox/{FAILED}/")
        return
    chat_store.record_sent(settings.chat_dir, settings.timezone, str(settings.telegram_chat_id), path,
                           photo_sent=photo is not None)
    log.info("Outbox message sent: %s%s; added to chat history", path.name, " with photo" if photo else "")


async def outbox_loop(telegram: TelegramClient, settings: Settings, stop: asyncio.Event,
                      telegram_health: Optional[Dependency] = None) -> None:
    log.info("outbox: watching %s", settings.chat_dir / chat_store.OUTBOX)
    outage = Outage("outbox", telegram_health)
    while not stop.is_set():
        wait = OUTBOX_POLL_SECONDS
        try:
            for f in chat_store.queued_messages(settings.chat_dir):
                if stop.is_set():
                    break
                await send_outbox_message(f, telegram, settings)
                outage.ok()
        except (TelegramError, *TRANSIENT) as e:
            waiting = len(chat_store.queued_messages(settings.chat_dir))
            wait = outage.failed(e, f"; {waiting} message(s) waiting")
        except Exception:  # noqa: BLE001
            log.exception("outbox loop iteration failed")
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=wait)


# ---------- incoming messages ----------

async def _keep_typing(telegram: TelegramClient) -> None:
    while True:
        await telegram.send_chat_action("typing")
        await asyncio.sleep(TYPING_EVERY_SECONDS)


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


async def _reply(telegram: TelegramClient, text: str) -> None:
    try:
        await telegram.send_message(text)
        log.info(f"  Reply sent: {text}")
    except (TelegramError, *TRANSIENT) as e:
        log.warning(f"  Reply not sent: {_describe(e)}")


async def security_mode_command(settings: Settings, telegram: TelegramClient, command: str) -> None:
    """/mode shows the security mode; /normal, /home, /away, /vacation set it."""
    if command == "mode":
        state = await asyncio.to_thread(security_mode.current, settings)
        log.info(f"  /mode command - security mode is {state.headline()}")
    else:
        state = await asyncio.to_thread(security_mode.set_mode, settings, command, "telegram")
        log.info(f"  /{command} command - security mode set to {state.headline()}")
    scheduled = await asyncio.to_thread(security_mode.upcoming, settings)
    await _reply(telegram, f"Security mode: {state.headline()}"
                 + "".join(f"\nScheduled: {e.describe()}" for e in scheduled))


async def handle_message(agent: ChatAgent, telegram: TelegramClient, chat_id: str, text: str) -> None:
    """One authorized message: "/clear", a security mode command, or the agent's
    reply (with "typing..." meanwhile)."""
    log.info(f"Message received from chat {chat_id}: {text}")
    command = text.split()[0].lower().lstrip("/").split("@")[0] if text.startswith("/") else ""
    if command in (*security_mode.MODES, "mode"):
        await security_mode_command(agent.settings, telegram, command)
        return
    if text.lower() == "/clear":
        log.info("  /clear command - clearing chat history...")
        prior = await agent.clear_history(chat_id)
        try:
            await telegram.send_message(f"cleared {prior} turn(s)")
            log.info("  Confirmation sent")
        except (TelegramError, *TRANSIENT) as e:
            log.warning(f"  Confirmation not sent: {_describe(e)}")
        return

    typing = asyncio.create_task(_keep_typing(telegram))
    t0 = time.perf_counter()
    try:
        reply = await agent.handle(chat_id, text)
    except Exception as e:  # noqa: BLE001 - the user gets the error as the reply: no separate alert
        log.exception(f"  Chat agent failed: {e} - error reply will be sent", extra={"alert": False})
        reply = f"(agent error: {type(e).__name__}: {e})"
    finally:
        await _cancel(typing)

    if not reply:
        log.info(f"  No reply sent; message handled in {time.perf_counter() - t0:.1f}s")
        return
    log.info("  Sending reply to Telegram...")
    try:
        await telegram.send_message(reply)
        log.info(f"  Reply sent in {time.perf_counter() - t0:.1f}s total: {reply}")
    except (TelegramError, *TRANSIENT) as e:
        log.warning(f"  Reply not sent: {_describe(e)}")


async def poll_loop(agent: ChatAgent, telegram: TelegramClient, authorized: str, stop: asyncio.Event,
                    health: Optional[Health] = None, telegram_health: Optional[Dependency] = None) -> None:
    offset: Optional[int] = None
    outage = Outage("incoming messages", telegram_health)
    while not stop.is_set():
        if health:
            health.ok()  # a pass of the loop: the bot runs (Telegram's own state is telegram_health)
        try:
            updates = await telegram.get_updates(offset=offset, timeout=30)
            outage.ok()
        except TelegramError as e:
            if e.permanent or e.code == 409:  # bad token / chat; 409: another bot polls with this token
                log.error(f"getUpdates failed: {e} - check TELEGRAM_BOT_TOKEN and that only one chat-bot runs")
                if telegram_health:
                    telegram_health.failed(f"refuses the bot: {e}")
                await asyncio.sleep(RETRY_MAX_SECONDS)
            else:
                await asyncio.sleep(outage.failed(e))
            continue
        except TRANSIENT as e:
            await asyncio.sleep(outage.failed(e))
            continue
        except Exception as e:  # noqa: BLE001
            log.exception("unexpected getUpdates error: %s", e)
            if health:
                health.failed(f"unexpected error: {_describe(e)}")
            await asyncio.sleep(RETRY_SECONDS)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            msg = update.get("message") or {}
            chat_id = str((msg.get("chat") or {}).get("id"))
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            if chat_id != authorized:
                log.warning(f"Message from unauthorized chat {chat_id} dropped")
                continue
            await handle_message(agent, telegram, chat_id, text)


# ---------- service ----------

async def serve(settings: Settings) -> None:
    telegram = TelegramClient(settings)
    agent = ChatAgent(settings, EventTools(settings, telegram))
    authorized = str(settings.telegram_chat_id)
    log.info(f"Chat bot started: authorized chat {authorized}; model {agent.model} effort "
             f"{agent.effort or 'default'}, memory changes {agent.memory_model} effort "
             f"{agent.memory_effort or 'default'} ({settings.llm_provider})")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):  # Windows: Ctrl+C raises KeyboardInterrupt instead
            loop.add_signal_handler(sig, stop.set)

    health, telegram_health = Health(settings, PROGRAM), Dependency(settings, "telegram")
    outbox = asyncio.create_task(outbox_loop(telegram, settings, stop, telegram_health))
    try:
        await poll_loop(agent, telegram, authorized, stop, health, telegram_health)
    finally:
        stop.set()
        await _cancel(outbox)
    log.info("chat bot stopped")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Also show the other modules' internal INFO logs")
    args = parser.parse_args(argv)
    setup_logging(PROGRAM, args.verbose, log.name, "blink_watcher.chat.agent", "blink_watcher.chat.tools")
    try:
        asyncio.run(serve(load_settings()))
    except KeyboardInterrupt:
        log.info("chat bot stopped (Ctrl+C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
