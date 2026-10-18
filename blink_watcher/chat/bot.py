"""The Telegram chat bot service.

Long-polls Telegram for incoming messages and hands them to the ChatAgent
(the same agent the notifier uses for incoming events). Only messages from
the authorized chat (TELEGRAM_CHAT_ID) are processed; everything else is
dropped. "/clear" empties today's chat history window; "/normal", "/home",
"/away" and "/vacation" set the security mode (core.security_mode), "/mode" shows it.

Also sends the messages the notifier queues in chat/outbox/: each file is one
message — text + the event's thumbnail as the photo — which is delivered to
Telegram, then recorded as sent (chat history + "sent_at" in the event list)
and removed. A network drop is logged in one line and retried; Ctrl+C stops.
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
from blink_watcher.core.logs import setup_logging
from blink_watcher.settings import Settings, load_settings

log = logging.getLogger("blink_watcher.chat.bot")

PROGRAM = "chat-bot"
OUTBOX_POLL_SECONDS = 2.0
RETRY_SECONDS = 5        # after a failed getUpdates
TYPING_EVERY_SECONDS = 4  # Telegram shows "typing..." for about 5 s


# ---------- outbox ----------

async def send_outbox_message(path: Path, telegram: TelegramClient, settings: Settings) -> None:
    """Send one chat/outbox message (text + photo) and record it as sent
    (chat.store.record_sent). On a Telegram error the file stays and is
    retried next tick."""
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
        if photo is not None:
            await telegram.send_photo(photo, caption=text or None)
        elif text:
            await telegram.send_message(text)
    except TelegramError as e:
        log.error("outbox: telegram send failed for %s: %s", path.name, e)
        return
    chat_store.record_sent(settings.chat_dir, settings.timezone, str(settings.telegram_chat_id), path,
                           photo_sent=photo is not None)
    log.info("Outbox message sent: %s%s; added to chat history", path.name, " with photo" if photo else "")


async def outbox_loop(telegram: TelegramClient, settings: Settings, stop: asyncio.Event) -> None:
    log.info("outbox: watching %s", settings.chat_dir / chat_store.OUTBOX)
    while not stop.is_set():
        try:
            for f in chat_store.queued_messages(settings.chat_dir):
                if stop.is_set():
                    break
                await send_outbox_message(f, telegram, settings)
        except Exception:  # noqa: BLE001
            log.exception("outbox loop iteration failed")
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=OUTBOX_POLL_SECONDS)


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
    except TelegramError as e:
        log.error(f"  Reply not sent: {e}")


async def security_mode_command(settings: Settings, telegram: TelegramClient, command: str) -> None:
    """/mode shows the security mode; /normal, /home, /away, /vacation set it."""
    if command == "mode":
        state = await asyncio.to_thread(security_mode.current, settings)
        log.info(f"  /mode command - security mode is {state.headline()}")
    else:
        state = await asyncio.to_thread(security_mode.set_mode, settings, command, "telegram")
        log.info(f"  /{command} command - security mode set to {state.headline()}")
    await _reply(telegram, f"Security mode: {state.headline()}")


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
        except TelegramError as e:
            log.error(f"  Confirmation not sent: {e}")
        return

    typing = asyncio.create_task(_keep_typing(telegram))
    t0 = time.perf_counter()
    try:
        reply = await agent.handle(chat_id, text)
    except Exception as e:  # noqa: BLE001
        log.exception(f"  Chat agent failed: {e} - error reply will be sent")
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
    except TelegramError as e:
        log.error(f"  Reply not sent: {e}")


async def poll_loop(agent: ChatAgent, telegram: TelegramClient, authorized: str, stop: asyncio.Event) -> None:
    offset: Optional[int] = None
    while not stop.is_set():
        try:
            updates = await telegram.get_updates(offset=offset, timeout=30)
        except TelegramError as e:
            log.error("getUpdates failed: %s", e)
            await asyncio.sleep(RETRY_SECONDS)
            continue
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            log.warning(f"Telegram unreachable, retrying in {RETRY_SECONDS}s: {e or type(e).__name__}")
            await asyncio.sleep(RETRY_SECONDS)
            continue
        except Exception as e:  # noqa: BLE001
            log.exception("unexpected getUpdates error: %s", e)
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

    outbox = asyncio.create_task(outbox_loop(telegram, settings, stop))
    try:
        await poll_loop(agent, telegram, authorized, stop)
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
