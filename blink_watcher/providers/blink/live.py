"""Blink live view: start a camera's liveview command and hand its stream to ffmpeg.

Blink answers the liveview request with a server URL: immis:// (Blink's own
protocol; blinkpy's BlinkLiveStream turns it into MPEG-TS, written to
ffmpeg's stdin) or rtsps:// (ffmpeg reads it itself).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from blinkpy import api
from blinkpy.blinkpy import Blink
from blinkpy.camera import BlinkCamera
from blinkpy.livestream import BlinkLiveStream

from blink_watcher.providers.base import LiveSource, ProviderError, UrlLiveSource

log = logging.getLogger(__name__)

# Blink's current liveview endpoints per camera type (as in blinkpy 0.25.9; the
# installed blinkpy's older ones get "An app update is required").
LIVEVIEW_PATHS = {
    "default": "/api/v6/accounts/{account}/networks/{network}/cameras/{camera}/liveview",
    "mini": "/api/v2/accounts/{account}/networks/{network}/owls/{camera}/liveview",
    "doorbell": "/api/v2/accounts/{account}/networks/{network}/doorbells/{camera}/liveview",
}


async def request_liveview(blink: Blink, camera: BlinkCamera) -> dict:
    """Ask Blink to start a camera's live view -> its answer ({"server": "immis://..." | "rtsps://...",
    "command_id", "polling_interval", ...}). Not waited on: the command runs as long as the stream."""
    path = LIVEVIEW_PATHS.get(camera.camera_type or "default", LIVEVIEW_PATHS["default"])
    url = blink.urls.base_url + path.format(account=blink.account_id, network=camera.network_id,
                                            camera=camera.camera_id)
    return await api.http_post(blink, url, data=json.dumps({"intent": "liveview", "motion_event_start_time": None}))


async def open_live_stream(blink: Blink, camera: BlinkCamera) -> LiveSource:
    response = await request_liveview(blink, camera)
    server = (response or {}).get("server") or ""
    if server.startswith("immis://"):
        return ImmisLiveSource(camera, response)
    if server.startswith("rtsp"):
        return UrlLiveSource(server, ["-rtsp_transport", "tcp"])
    raise ProviderError(f"Blink did not start a live view ({response})")


class ImmisLiveSource(LiveSource):
    """An immis:// live view: ImmisStream writes its MPEG-TS video to ffmpeg's stdin."""
    kind = "immis"
    input_args = ["-f", "mpegts", "-i", "pipe:0"]
    uses_stdin = True

    def __init__(self, camera: BlinkCamera, response: dict):
        self.stream = ImmisStream(camera, response)
        self._feeder: Optional[asyncio.Task] = None

    async def start(self, stdin: Optional[asyncio.StreamWriter]) -> None:
        self.stream.clients.append(stdin)  # the MPEG-TS goes straight into ffmpeg
        self._feeder = asyncio.create_task(self.stream.feed())

    async def stop(self) -> None:
        if self._feeder:
            self._feeder.cancel()


class ImmisStream(BlinkLiveStream):
    """blinkpy's BlinkLiveStream, with the string fields of its auth header
    filled in (blinkpy 0.25 leaves them out): a 4-byte length, then the
    string's bytes padded to that length; and a receiver that reads whole
    packets (blinkpy's read(n) can return less and then gives up) and logs
    what Blink sent."""

    def add_auth_header_string_field(self, auth_header, field_string, max_length):
        value = str(field_string or "")
        if len(value) != max_length:
            log.info(f"Live view auth field is {len(value)} chars, padded to {max_length}")
        auth_header.extend(max_length.to_bytes(4, byteorder="big"))
        auth_header.extend(value.encode("utf-8")[:max_length].ljust(max_length, b"\x00"))
        return auth_header

    async def recv(self):
        packets: dict[int, int] = {}
        video = 0
        try:
            while True:
                header = await self.target_reader.readexactly(9)
                length = int.from_bytes(header[5:9], byteorder="big")
                payload = await self.target_reader.readexactly(length) if length > 0 else b""
                packets[header[0]] = packets.get(header[0], 0) + 1
                if header[0] != 0x00 or payload[:1] != b"\x47":  # only MPEG-TS video
                    continue
                video += len(payload)
                for writer in self.clients:
                    if not writer.is_closing():
                        writer.write(payload)
                        await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        except Exception as e:  # noqa: BLE001
            log.warning(f"Live view receive failed: {type(e).__name__}: {e}")
        finally:
            log.info(f"Live view data from Blink ended: {video} video bytes; packets by type {packets}")
            self.target_writer.close()

    async def poll(self):
        """Keep the liveview command polled while the stream runs. blinkpy's
        poll ends the command (and so the stream) on any state it does not
        expect; here the stream ends only when Blink closes it."""
        blink, last = self.camera.sync.blink, None
        try:
            while not self.target_reader.at_eof():
                try:
                    status = await api.request_command_status(blink, self.camera.network_id, self.command_id)
                    state = next((c.get("state_condition") for c in (status or {}).get("commands", [])
                                  if c.get("id") == self.command_id), (status or {}).get("status_code"))
                    if state != last:
                        log.info(f"Live view command state: {state}")
                        last = state
                except Exception as e:  # noqa: BLE001
                    log.debug(f"Live view command poll failed: {e}")
                await asyncio.sleep(self.polling_interval or 5)
        finally:
            try:
                await api.request_command_done(blink, self.camera.network_id, self.command_id)
            except Exception:  # noqa: BLE001
                pass
