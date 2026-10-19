"""The configuration page (settings.document: the options documented in
.env.example with their .env values).

  GET  /api/settings    groups and options with their .env values (secrets only as set / not set),
                        "writable": whether this client may save
  POST /api/settings    {"changes": {KEY: value | null}} -> validated, written to .env
                        (from this PC only; null = back to the default); "restart": the
                        services that read them, "auto_restart": run's supervisor restarts them
"""
from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import web

from blink_watcher.settings import document
from blink_watcher.supervisor import SUPERVISED

log = logging.getLogger(__name__)

routes = web.RouteTableDef()


def _local(request: web.Request) -> bool:
    return (request.remote or "") in {"127.0.0.1", "::1", "localhost"}


@routes.get("/api/settings")
async def settings_get(request: web.Request) -> web.Response:
    data = await asyncio.to_thread(document.options)
    return web.json_response({**data, "writable": _local(request)})


@routes.post("/api/settings")
async def settings_post(request: web.Request) -> web.Response:
    if not _local(request):
        raise web.HTTPForbidden(text="configuration can only be changed from this PC")
    body = await request.json()
    changes = body.get("changes") or {}
    if not isinstance(changes, dict) or not changes:
        raise web.HTTPBadRequest(text="no changes")
    result = await asyncio.to_thread(document.update, changes)
    if result.get("ok"):
        result["auto_restart"] = os.environ.get(SUPERVISED) == "1"  # run's supervisor applies it
    if result.get("ok"):
        log.info(f"Configuration saved: {', '.join(result['changed'])} "
                 f"(restart: {', '.join(result['restart']) or 'none'})")
    else:
        log.warning(f"Configuration not saved: {result.get('errors')}")
    return web.json_response(result, status=200 if result.get("ok") else 422)
