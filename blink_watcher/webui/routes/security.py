"""The security mode (core.security_mode) in the console's header.

  GET  /api/security-mode    {"state": {mode, label, since, until, by, note}, "headline", "modes": [{value, label}],
                           "home_until": HOME_MODE_UNTIL, "schedule": [{id, mode, start, end, note, text}]}
  POST /api/security-mode    {"mode": normal | home | away | vacation} -> the new state (as GET)
"""
from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from blink_watcher.core import security_mode
from blink_watcher.webui.context import context

log = logging.getLogger(__name__)

routes = web.RouteTableDef()


def _body(settings, state: security_mode.ModeState) -> dict:
    return {"state": state.to_dict(), "headline": state.headline(),
            "modes": [{"value": m, "label": security_mode.LABELS[m]} for m in security_mode.MODES],
            "home_until": settings.home_mode_until,
            "schedule": [{"id": e.id, "mode": e.mode, "start": e.start, "end": e.end, "note": e.note,
                          "text": e.describe()} for e in security_mode.upcoming(settings)]}


@routes.get("/api/security-mode")
async def get_mode(request: web.Request) -> web.Response:
    settings = context(request).settings
    return web.json_response(_body(settings, await asyncio.to_thread(security_mode.current, settings)))


@routes.post("/api/security-mode")
async def set_mode(request: web.Request) -> web.Response:
    settings = context(request).settings
    try:
        mode = (await request.json()).get("mode")
        state = await asyncio.to_thread(security_mode.set_mode, settings, mode, "console")
    except (ValueError, AttributeError) as e:
        raise web.HTTPBadRequest(text=str(e))
    return web.json_response(_body(settings, state))
