"""Health of the services and of the outside services they use (core.health),
for monitoring and the console's warning banner.

  GET /api/health    200 when every module is OK, 503 when one is DOWN:
      {"status": "OK" | "DOWN", "checked_at",
       "modules": [{name, type: "service" | "dependency", label, status: "OK" | "DOWN",
                    description, last_report_at}]}

Services: fetch, process, notify, chat-bot (the web UI is the one answering).
Dependencies: blink (fetch's polls), telegram (the chat-bot's), ai (every LLM
call). A module that has not reported yet is assumed up. A service is DOWN
when it reports failures or its last report is older than
HEALTH_STALE_MINUTES (stopped or stuck); a dependency only when it was last
seen failing.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from aiohttp import web

from blink_watcher.core.health import DEPENDENCIES, LABELS, SERVICES, read_all
from blink_watcher.webui.context import context

routes = web.RouteTableDef()

SEEN_AS = {"blink": "poll", "telegram": "contact", "ai": "call"}  # what a report of the dependency is


def _when(value: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def _at(t: Optional[datetime], now: datetime) -> str:
    """HH:MM:SS today, else MM-DD HH:MM."""
    if t is None:
        return "?"
    t = t.astimezone(now.tzinfo)
    return f"{t:%H:%M:%S}" if t.date() == now.date() else f"{t:%m-%d %H:%M}"


def _module(name: str, h: Optional[dict], now: datetime, stale: int, provider: str) -> dict:
    kind = "dependency" if name in DEPENDENCIES else "service"
    label = (h or {}).get("label") or LABELS.get(name) or ("AI (%s)" % provider if name == "ai" else name)
    updated = _when((h or {}).get("updated_at"))
    seen = SEEN_AS.get(name)

    def out(status: str, description: str) -> dict:
        return {"name": name, "type": kind, "label": label, "status": status, "description": description,
                "last_report_at": updated.isoformat(timespec="seconds") if updated else None}

    # No report yet (just started, or never used): assumed up. A dependency's
    # old report is fine too: its reporting service shows DOWN if that one stopped.
    if h is None or updated is None:
        return out("OK", "No report yet; assumed up")
    if kind == "service" and stale > 0 and now - updated > timedelta(minutes=stale):
        return out("DOWN", f"Not running: no report since {_at(updated, now)}")
    if not h.get("ok", True):
        failures = int(h.get("failures") or 0)
        count = f" ({failures} failures in a row)" if failures > 1 else ""
        error = f": {h['error']}" if h.get("error") else ""
        return out("DOWN", f"Failing since {_at(_when(h.get('failing_since')), now)}{count}{error}")
    last_ok = _when(h.get("last_ok_at"))
    if seen:
        if last_ok is None:
            return out("OK", f"No {seen} yet")
        return out("OK", f"Last {seen} {_at(last_ok, now)} succeeded")
    return out("OK", f"Working; last report {_at(updated, now)}")


def status(settings) -> dict:
    now = datetime.now(ZoneInfo(settings.timezone))
    states = read_all(settings)
    modules = [_module(name, states.get(name), now, settings.health_stale_minutes, settings.llm_provider)
               for name in (*SERVICES, *DEPENDENCIES)]
    return {"status": "DOWN" if any(m["status"] == "DOWN" for m in modules) else "OK",
            "checked_at": now.isoformat(timespec="seconds"), "modules": modules}


@routes.get("/api/health")
async def get_health(request: web.Request) -> web.Response:
    settings = context(request).settings
    body = await asyncio.to_thread(status, settings)
    return web.json_response(body, status=200 if body["status"] == "OK" else 503)
