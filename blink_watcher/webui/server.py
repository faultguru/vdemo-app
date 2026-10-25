"""Web UI: a local monitoring console over the chat area's daily event lists,
with a live monitor, a chat with the analyst and the configuration page.

    python -m blink_watcher webui                       # http://127.0.0.1:9090 (WEBUI_HOST / WEBUI_PORT)
    python -m blink_watcher webui --host 0.0.0.0 --port 9000

Serves two single-page apps from webui/static — / (index.html: events,
activities, incoming clips, live monitor, chat) and /settings (settings.html),
the faultguru logo (/logo.svg, /favicon.ico; made by brand/make_brand.py) —
and their JSON API, one module per area in webui.routes (each lists its routes):
events, incoming, settings, live, chat.

Paths in the API are relative to STORAGE_ROOT, as stored in the events
("videos.processed/..."). Only the media areas are served (webui.context); nothing
is written but the live monitor's snapshots and, from this PC only, .env.
"""
from __future__ import annotations

import argparse
import logging
from importlib.resources import files
from typing import Optional

from aiohttp import web

from blink_watcher.core.logs import setup_logging
from blink_watcher.settings import Settings, load_settings
from blink_watcher.settings import get_settings
from blink_watcher.webui.context import CONTEXT, WebContext
from blink_watcher.webui.routes import chat, events, health, incoming, live, security, settings as settings_routes

_settings = get_settings()

log = logging.getLogger("blink_watcher.webui.server")

PROGRAM = "webui"
STATIC = files("blink_watcher.webui") / "static"
MAX_UPLOAD_MB = _settings.webui_max_upload_mb  # files sent with a chat message


def _page(name: str):
    async def handler(request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC / name, headers={"Cache-Control": "no-cache"})
    return handler


def make_app(settings: Settings) -> web.Application:
    app = web.Application(client_max_size=MAX_UPLOAD_MB * 2**20)
    ctx = app[CONTEXT] = WebContext(settings)
    live.setup(app, settings)
    app[chat.CONSOLE] = chat.Console(ctx)
    app.add_routes([web.get("/", _page("index.html")), web.get("/settings", _page("settings.html")),
                    web.get("/logo.svg", _page("logo.svg")), web.get("/favicon.ico", _page("favicon.ico"))])
    for module in (events, incoming, settings_routes, live, chat, security, health):
        app.add_routes(module.routes)
    return app


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=None, help="Override WEBUI_HOST")
    parser.add_argument("--port", type=int, default=None, help="Override WEBUI_PORT")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also log every HTTP request")
    args = parser.parse_args(argv)
    log_file = setup_logging(PROGRAM, args.verbose, "blink_watcher.webui", "blink_watcher.chat.agent",
                             "blink_watcher.chat.tools", "blink_watcher.live", "blink_watcher.providers")

    settings = load_settings()
    host, port = args.host or settings.webui_host, args.port or settings.webui_port
    log.info(f"Web UI started: http://{host}:{port}/ (storage root {settings.storage_path}, log file {log_file})")
    web.run_app(make_app(settings), host=host, port=port, print=None,
                access_log=logging.getLogger("aiohttp.access") if args.verbose else None)
    log.info("Web UI stopped")
    return 0
