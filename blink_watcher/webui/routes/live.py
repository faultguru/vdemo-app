"""The live monitor (live.LiveMonitor): the cameras' latest snapshots and their
true live view.

  GET /api/live/cameras            the enabled cameras on the provider account
  GET /api/live/status?cam=..      heartbeat of the live monitor for those cameras -> their latest
                                   snapshots; keeps the snapshot loop going
  GET /api/live/stream/<camera>    a camera's live view as MJPEG (multipart/x-mixed-replace)

The monitor's snapshots are the only files the web UI writes (LIVE_SNAPSHOTS area).
"""
from __future__ import annotations

import time
from typing import Optional

from aiohttp import web

from blink_watcher.live.monitor import LiveError, LiveMonitor
from blink_watcher.settings import Settings

routes = web.RouteTableDef()


class Live:
    def __init__(self, settings: Settings):
        self.monitor = LiveMonitor(settings)
        self._cameras: Optional[list[str]] = None

    async def cameras(self) -> list[str]:
        """The provider's cameras, asked once (the first call connects)."""
        if self._cameras is None:
            try:
                self._cameras = await self.monitor.cameras()
            except LiveError as e:
                raise web.HTTPServiceUnavailable(text=str(e))
        return self._cameras


LIVE = web.AppKey("live", Live)


def setup(app: web.Application, settings: Settings) -> Live:
    live = app[LIVE] = Live(settings)

    async def close(app: web.Application) -> None:
        await live.monitor.close()

    app.on_cleanup.append(close)
    return live


@routes.get("/api/live/cameras")
async def live_cameras(request: web.Request) -> web.Response:
    return web.json_response({"cameras": await request.app[LIVE].cameras()})


@routes.get("/api/live/status")
async def live_status(request: web.Request) -> web.Response:
    live = request.app[LIVE]
    known = set(await live.cameras())
    return web.json_response(live.monitor.status([c for c in request.query.getall("cam", []) if c in known]))


@routes.get("/api/live/stream/{camera}")
async def live_stream(request: web.Request) -> web.StreamResponse:
    live = request.app[LIVE]
    name = request.match_info["camera"]
    if name not in await live.cameras():
        raise web.HTTPNotFound(text=f"unknown camera {name}")
    stream = live.monitor.stream(name)
    stream.viewers += 1
    try:
        await stream.started.wait()
        if stream.error and not stream.seq:
            raise web.HTTPServiceUnavailable(text=stream.error)
        resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                           "Cache-Control": "no-store"})
        await resp.prepare(request)
        seq = 0
        while True:
            frame, seq = await stream.next_frame(seq)
            if frame is None:
                break
            await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
        return resp
    except (ConnectionResetError, ConnectionError):
        return web.Response(status=499)
    finally:
        stream.viewers -= 1
        stream.left_at = time.monotonic()
