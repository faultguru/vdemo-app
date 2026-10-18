"""The console's event views: configuration for the page, the days, a day's
events (and activities), the camera list, a clip's .metadata.json and media files.

  GET /api/config                  today (local date), timezone, poll interval, stitching on/off,
                                   the enabled cameras' names
  GET /api/cameras                 every known camera (webui.cameras) with its enabled flag and
                                   detection filter
  GET /api/days                    dates with an events list (chat/<date>/events.json), newest first
  GET /api/events/<YYYY-MM-DD>     that day's events, oldest first, and with STITCH_ENABLED its
                                   activities (stitch.Stitcher, newest first)
  GET /api/metadata?path=<rel>     a .metadata.json under STORAGE_ROOT
  GET /media/<rel>                 a video or picture under STORAGE_ROOT (Range requests: playback
                                   starts at once, seeking works)
"""
from __future__ import annotations

import json
import logging

from aiohttp import web

from blink_watcher.chat import store as chat_store
from blink_watcher.webui.context import context, date_param

log = logging.getLogger(__name__)

MEDIA_SUFFIXES = {".mp4", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".webm", ".mov"}

routes = web.RouteTableDef()


@routes.get("/api/config")
async def config(request: web.Request) -> web.Response:
    ctx = context(request)
    return web.json_response({
        "today": ctx.today(),
        "timezone": ctx.settings.timezone,
        "poll_sec": ctx.settings.webui_poll_sec,
        "stitch": ctx.stitcher is not None,
        "cameras": [c["name"] for c in ctx.cameras() if c["enabled"]],
    })


@routes.get("/api/cameras")
async def cameras(request: web.Request) -> web.Response:
    return web.json_response({"cameras": context(request).cameras()})


@routes.get("/api/days")
async def days(request: web.Request) -> web.Response:
    ctx = context(request)
    out = []
    for date in sorted(chat_store.event_dates(ctx.chat_dir), reverse=True):
        events = chat_store.load_events(ctx.chat_dir, date)
        out.append({
            "date": date,
            "events": len(events),
            "sent": sum(1 for e in events if (e.get("notification") or {}).get("decision") == "send"),
            "threats": sum(1 for e in events if ((e.get("summary") or {}).get("threat") or {}).get("level")
                           not in (None, "none")),
        })
    return web.json_response({"today": ctx.today(), "days": out})


@routes.get("/api/events/{date}")
async def events(request: web.Request) -> web.Response:
    ctx, date = context(request), date_param(request)
    day_events = chat_store.load_events(ctx.chat_dir, date)
    groups = None
    if ctx.stitcher:
        try:
            groups = ctx.stitcher.day(date, day_events)
        except Exception:  # noqa: BLE001 — the events still show, unstitched
            log.exception(f"Stitching {date} failed")
    return web.json_response({"date": date, "today": ctx.today(), "events": day_events, "groups": groups})


@routes.get("/api/metadata")
async def metadata(request: web.Request) -> web.Response:
    path = context(request).resolve(request.query.get("path", ""), {".metadata.json"})
    try:
        return web.json_response(json.loads(path.read_text(encoding="utf-8")))
    except ValueError:
        raise web.HTTPUnprocessableEntity(text="metadata file is not valid JSON")


@routes.get("/media/{rel:.+}")
async def media(request: web.Request) -> web.StreamResponse:
    path = context(request).resolve(request.match_info["rel"], MEDIA_SUFFIXES)
    return web.FileResponse(path, headers={"Cache-Control": "max-age=3600"})
