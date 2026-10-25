"""The console chat: its own chat agent (chat.agent.ChatAgent in console mode),
not connected to the Telegram bot.

  GET  /api/chat          the console chat so far
  POST /api/chat          a message (multipart: text, date?, live?, files...) -> the agent's
                          reply + the pictures / videos it sent
  POST /api/chat/clear    start the console chat over

The conversation is kept in memory only (gone on restart); the agent reads the
bot's long-term memory but cannot change it, and does not see or write the
bot's day notes. The pictures / videos it sends come back to the browser.
With a date selected, questions are about that day (beyond it: the last
WEBUI_CHAT_LOOKBACK_DAYS days); without, the lookback period. Pictures the
user sends go to the agent (in memory, never stored); for a video, a few
frames do. With the live monitor open, the message names the cameras it shows
and the agent also gets their latest snapshots (WEBUI_LIVE_AI_FRAMES per camera
from the last WEBUI_LIVE_AI_WINDOW_SEC) for questions about now.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from aiohttp import web

from blink_watcher.ai import ImagePart, Part, TextPart
from blink_watcher.chat.agent import ChatAgent
from blink_watcher.chat.prompts import live_block
from blink_watcher.chat.tools import EventTools
from blink_watcher.core.images import ImageError, fit_jpeg
from blink_watcher.settings import get_settings
from blink_watcher.webui.context import DATE, WebContext, context
from blink_watcher.webui.routes.live import LIVE

_settings = get_settings()

log = logging.getLogger(__name__)

CHAT_ID = "webui"
TRANSCRIPT_MAX = _settings.webui_chat_transcript_max  # console messages kept in memory for page reloads
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov"}
VIDEO_FRAMES = _settings.webui_chat_video_frames      # frames of an uploaded video shown to the agent

routes = web.RouteTableDef()


class WebMedia:
    """Delivery target for the agent's send_thumbnail / send_video in the
    console: collects them (as paths under STORAGE_ROOT) for the reply."""

    def __init__(self, root: Path):
        self.root = root
        self.items: list[dict] = []

    def _add(self, kind: str, path: Path, caption: Optional[str]) -> dict:
        rel = Path(path).resolve().relative_to(self.root).as_posix()
        self.items.append({"kind": kind, "path": rel, "caption": caption or ""})
        return {"ok": True}

    async def send_photo(self, photo: Path, caption: Optional[str] = None) -> dict:
        return self._add("photo", photo, caption)

    async def send_video(self, video: Path, caption: Optional[str] = None) -> dict:
        return self._add("video", video, caption)


class Console:
    def __init__(self, ctx: WebContext):
        self.sink = WebMedia(ctx.root)
        self.agent = ChatAgent(ctx.settings, EventTools(ctx.settings, self.sink), console=True)
        self.lock = asyncio.Lock()        # one console message at a time (the sink is shared)
        self.transcript: list[dict] = []  # what the console shows; memory only


CONSOLE = web.AppKey("console", Console)


# ---------- the user's files ----------

async def _read_upload(part) -> tuple[dict, list[Part]]:
    """A file the user sent -> (what the console shows, parts for the agent).
    Read in memory; never stored."""
    name = Path(part.filename or "file").name
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise web.HTTPBadRequest(text=f"unsupported file type: {name} (pictures and videos only)")
    data = await part.read(decode=False)
    if suffix in VIDEO_SUFFIXES:
        return {"kind": "video", "name": name}, await asyncio.to_thread(_video_frames, data, suffix)
    try:
        parts: list[Part] = [ImagePart(fit_jpeg(data))]
    except ImageError:
        parts = [TextPart(f"(picture {name} could not be read)")]
    return {"kind": "photo", "name": name}, parts


def _video_frames(data: bytes, suffix: str) -> list[Part]:
    """A few evenly spaced frames of an uploaded video (OpenCV needs a file:
    a temporary one, deleted right after)."""
    import cv2

    parts: list[Part] = []
    with tempfile.TemporaryDirectory(prefix="webui-upload-", ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        cap = cv2.VideoCapture(str(path))
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), cap.get(cv2.CAP_PROP_FPS) or 0
        for i in range(VIDEO_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(count * (i + 0.5) / VIDEO_FRAMES))
            ok, frame = cap.read()
            if not ok:
                continue
            ok, jpg = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            at = f" at {count * (i + 0.5) / VIDEO_FRAMES / fps:.1f}s" if fps else ""
            parts += [TextPart(f"Frame {i + 1} of the video the user sent{at}:"), ImagePart(fit_jpeg(jpg.tobytes()))]
        cap.release()
    return parts or [TextPart("(the video the user sent could not be read)")]


# ---------- routes ----------

@routes.get("/api/chat")
async def chat_history(request: web.Request) -> web.Response:
    return web.json_response({"lookback_days": context(request).settings.webui_chat_lookback_days,
                              "messages": request.app[CONSOLE].transcript})


@routes.post("/api/chat/clear")
async def chat_clear(request: web.Request) -> web.Response:
    console = request.app[CONSOLE]
    async with console.lock:
        console.transcript = []
        await console.agent.clear_history(CHAT_ID)
    log.info("Console chat cleared")
    return web.json_response({"ok": True})


@routes.post("/api/chat")
async def chat(request: web.Request) -> web.Response:
    ctx, console = context(request), request.app[CONSOLE]
    settings = ctx.settings
    text, focus, uploads, attachments, live = "", None, [], [], []
    async for part in await request.multipart():
        if part.name == "text":
            text = (await part.text()).strip()
        elif part.name == "date":
            value = (await part.text()).strip()
            focus = value if DATE.match(value) else None
        elif part.name == "live":
            live = [c for c in (await part.text()).split("\n") if c.strip()]
        elif part.name == "files" and part.filename:
            media, parts = await _read_upload(part)
            uploads.append(media)
            attachments += parts
    if not text and not uploads:
        raise web.HTTPBadRequest(text="empty message")
    if uploads and not text:
        text = "(sent without text)"
    log.info(f"Console message received{f' (focus {focus})' if focus else ' (no focus date)'}: {text}"
             + (f" + {len(uploads)} file(s)" if uploads else ""))
    user_entry = {"role": "user", "text": text, "date": focus, "media": uploads, "live": live, "at": ctx.now_iso()}

    live_text, live_images = "", []
    if live:
        cams, live_images = await asyncio.to_thread(request.app[LIVE].monitor.ai_context, live,
                                                    settings.webui_live_ai_frames, settings.webui_live_ai_window_sec)
        live_text = live_block(cams, datetime.now(ctx.tz).strftime("%H:%M:%S"))
        log.info(f"Live monitor context: {len(live)} camera(s), "
                 f"{sum(isinstance(p, ImagePart) for p in live_images)} snapshot(s)")

    t0 = time.perf_counter()
    async with console.lock:
        console.sink.items = []
        try:
            reply = await console.agent.handle(CHAT_ID, text, focus_date=focus,
                                               lookback_days=settings.webui_chat_lookback_days,
                                               attachments=attachments, context=live_text,
                                               context_images=live_images)
        except Exception as e:  # noqa: BLE001
            log.exception("Console chat failed")
            reply = f"(agent error: {type(e).__name__}: {e})"
        media = list(console.sink.items)
    agent_entry = {"role": "agent", "text": reply, "media": media, "at": ctx.now_iso()}
    console.transcript = (console.transcript + [user_entry, agent_entry])[-TRANSCRIPT_MAX:]
    log.info(f"Console reply sent in {time.perf_counter() - t0:.1f}s: {len(media)} media item(s); "
             f"{reply[:160] if reply else '(no text)'}")
    return web.json_response({"user": user_entry, "agent": agent_entry})
