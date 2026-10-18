"""GET /api/incoming/<YYYY-MM-DD>: that day's clips still in the pipeline
(queued, processing, awaiting notification, just discarded) — webui.incoming."""
from __future__ import annotations

import asyncio

from aiohttp import web

from blink_watcher.webui.context import context, date_param
from blink_watcher.webui.incoming import incoming as incoming_clips

routes = web.RouteTableDef()


@routes.get("/api/incoming/{date}")
async def incoming(request: web.Request) -> web.Response:
    ctx, date = context(request), date_param(request)
    items = await asyncio.to_thread(incoming_clips, ctx.settings, date)
    return web.json_response({"date": date, "items": items})
