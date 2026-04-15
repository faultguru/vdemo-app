"""Standalone Telegram chat-bot entrypoint.

Long-polls Telegram for incoming messages and dispatches them to the
Gemini-backed ``ChatAgent``. Only messages from the authorized chat_id
(``TELEGRAM_CHAT_ID``) are processed; everything else is dropped.

Also drains server-initiated messages from ``events/outbox/*.json``:
each file is a single message (optionally with a photo path), which is
delivered to Telegram, appended to chat history, and then deleted.

Run with: ``python -m blink_watcher.chat_bot``
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
from pathlib import Path

from .chat_agent import ChatAgent
from .config import load_settings
from .event_tools import EventTools, EventToolError
from .telegram import TelegramClient, TelegramError

log = logging.getLogger("blink_watcher.chat_bot")

OUTBOX_POLL_SECONDS = 2.0


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


async def _process_outbox_file(
    path: Path,
    telegram: TelegramClient,
    tools: EventTools,
    agent: ChatAgent,
    chat_id: str,
) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        log.warning("outbox: bad file %s (%s); deleting", path.name, e)
        path.unlink(missing_ok=True)
        return

    model_prompt = (data.get("prompt") or "").strip()
    event_id = data.get("event_id") or path.stem
    if not model_prompt:
        log.warning("outbox: empty message %s; deleting", path.name)
        path.unlink(missing_ok=True)
        return

    photo_path: Path | None = None
    if event_id:
        try:
            ev = tools.get_event_details(event_id, event_id[:10])
            thumb_rel = ev.get("thumbnail")
            if thumb_rel:
                photo_path = tools.resolve_under_events(thumb_rel)
        except EventToolError as e:
            log.warning("outbox: %s could not resolve thumbnail: %s", path.name, e)

    user_text = ""
    if model_prompt:
        try:
            user_text = await agent.compose_notification(model_prompt)
        except Exception:  # noqa: BLE001
            log.exception("outbox: compose_notification failed for %s", path.name)
            return  # keep file, retry next tick
        log.info("outbox: %s model response: %s", path.name, user_text)

    if user_text.strip().upper().startswith("SKIP"):
        log.info("outbox: %s suppressed — %s", path.name, user_text.strip())
        path.unlink(missing_ok=True)
        return

    stripped = user_text.lstrip()
    for tag in ("<cautious>", "<friendly>", "<informative>"):
        if stripped.lower().startswith(tag):
            user_text = stripped[len(tag):].lstrip()
            break

    try:
        if photo_path is not None:
            await telegram.send_photo(photo_path, caption=user_text or None)
        elif user_text:
            await telegram.send_message(user_text)
    except TelegramError as e:
        log.error("outbox: telegram send failed for %s: %s", path.name, e)
        return  # keep the file so it retries next tick

    if user_text:
        try:
            await agent.append_notification(chat_id, user_text, event_id)
        except Exception:  # noqa: BLE001
            log.exception("outbox: append_notification failed for %s", path.name)

    path.unlink(missing_ok=True)
    log.info("outbox: delivered %s", path.name)


async def _outbox_loop(
    outbox_dir: Path,
    telegram: TelegramClient,
    tools: EventTools,
    agent: ChatAgent,
    chat_id: str,
    stop: asyncio.Event,
) -> None:
    outbox_dir.mkdir(parents=True, exist_ok=True)
    log.info("outbox: watching %s", outbox_dir)
    while not stop.is_set():
        try:
            files = sorted(outbox_dir.glob("*.json"))
            for f in files:
                if stop.is_set():
                    break
                await _process_outbox_file(f, telegram, tools, agent, chat_id)
        except Exception:  # noqa: BLE001
            log.exception("outbox loop iteration failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=OUTBOX_POLL_SECONDS)
        except asyncio.TimeoutError:
            pass


async def _run() -> None:
    settings = load_settings()
    telegram = TelegramClient(settings)
    tools = EventTools(settings, telegram)
    agent = ChatAgent(settings, tools)

    authorized = str(settings.telegram_chat_id)
    outbox_dir = settings.events_dir / "outbox"
    log.info("chat bot started; authorized chat_id=%s model=%s",
             authorized, settings.gemini_chat_model)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows

    outbox_task = asyncio.create_task(
        _outbox_loop(outbox_dir, telegram, tools, agent, authorized, stop)
    )

    offset: int | None = None
    try:
        while not stop.is_set():
            try:
                updates = await telegram.get_updates(offset=offset, timeout=30)
            except TelegramError as e:
                log.error("getUpdates failed: %s", e)
                await asyncio.sleep(5)
                continue
            except Exception as e:  # noqa: BLE001
                log.exception("unexpected getUpdates error: %s", e)
                await asyncio.sleep(5)
                continue

            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat = msg.get("chat") or {}
                incoming_id = str(chat.get("id"))
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                if incoming_id != authorized:
                    log.warning("dropping message from unauthorized chat_id=%s", incoming_id)
                    continue

                log.info("user: %s", text)

                if text.strip().lower() == "/clear":
                    prior = await agent.clear_history(incoming_id)
                    try:
                        await telegram.send_message(f"cleared {prior} turn(s)")
                    except TelegramError as e:
                        log.error("send_message failed: %s", e)
                    continue

                async def _typing_loop() -> None:
                    while True:
                        try:
                            await telegram.send_chat_action("typing")
                        except Exception:  # noqa: BLE001
                            pass
                        await asyncio.sleep(4)

                typing_task = asyncio.create_task(_typing_loop())
                try:
                    reply = await agent.handle(incoming_id, text)
                except Exception as e:  # noqa: BLE001
                    log.exception("agent failed")
                    reply = f"(agent error: {type(e).__name__}: {e})"
                finally:
                    typing_task.cancel()
                    try:
                        await typing_task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass

                if not reply:
                    log.info("bot: (silent — no follow-up after media send)")
                else:
                    try:
                        await telegram.send_message(reply)
                        log.info("bot: %s", reply[:200])
                    except TelegramError as e:
                        log.error("send_message failed: %s", e)
    finally:
        stop.set()
        outbox_task.cancel()
        try:
            await outbox_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    log.info("chat bot stopped")


def cli() -> None:
    _setup_logging()
    asyncio.run(_run())


if __name__ == "__main__":
    cli()
