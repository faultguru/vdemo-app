import argparse
import asyncio
import logging
import shutil
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import blink_api
from .auth import AuthError, TwoFARequired, connect
from .backfill import run_backfill
from .config import load_settings
from .downloader import Downloader
from .event_log import process_video
from .poller import LivePoller
from .store import Store
from .telegram import TelegramClient, TelegramError


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("blinkpy").setLevel(logging.DEBUG)


async def _send_telegram(text: str, markdown: bool) -> None:
    settings = load_settings()
    try:
        client = TelegramClient(settings)
        await client.send_message(text, parse_mode="MarkdownV2" if markdown else None)
        logging.info("telegram: sent (%d chars)", len(text))
    except TelegramError as e:
        logging.error("%s", e)
        raise SystemExit(1)


async def _run_live() -> None:
    settings = load_settings()
    store = Store(settings.state_dir / "state.db")
    try:
        blink = await connect(settings)
    except TwoFARequired as e:
        logging.error("%s", e)
        raise SystemExit(2)
    except AuthError as e:
        logging.error("%s", e)
        raise SystemExit(1)

    poller = LivePoller(settings, store, blink)
    downloader = Downloader(settings, store, blink)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _signal() -> None:
        stop.set()
        poller.stop()
        downloader.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal)
        except NotImplementedError:
            pass  # Windows

    try:
        await asyncio.gather(poller.run(), downloader.run())
    finally:
        await blink.auth.session.close()


async def _drain_queue(settings, store, blink) -> None:
    downloader = Downloader(settings, store, blink)
    task = asyncio.create_task(downloader.run())
    log = logging.getLogger("blink_watcher.drain")
    try:
        stalled_ticks = 0
        last_remaining = None
        while True:
            await asyncio.sleep(2)
            counts = store.count_by_state()
            pending = counts.get("pending", 0) + counts.get("claimed", 0)
            log.info("queue: %s", counts)
            if pending == 0:
                log.info("queue drained")
                break
            if pending == last_remaining:
                stalled_ticks += 1
                if stalled_ticks >= 15:
                    log.warning("no progress for 30s, aborting drain with %d pending", pending)
                    break
            else:
                stalled_ticks = 0
            last_remaining = pending
    finally:
        downloader.stop()
        await task


async def _run_backfill(hours: int, restart: bool, dry_run: bool, then_live: bool) -> None:
    settings = load_settings()
    store = Store(settings.state_dir / "state.db")
    try:
        blink = await connect(settings)
    except TwoFARequired as e:
        logging.error("%s", e)
        raise SystemExit(2)
    except AuthError as e:
        logging.error("%s", e)
        raise SystemExit(1)
    try:
        if then_live:
            downloader = Downloader(settings, store, blink)
            drain_task = asyncio.create_task(downloader.run())
            try:
                await run_backfill(settings, store, blink, hours=hours, restart=restart, dry_run=dry_run)
                poller = LivePoller(settings, store, blink)
                await poller.run()
            finally:
                downloader.stop()
                await drain_task
        else:
            await run_backfill(settings, store, blink, hours=hours, restart=restart, dry_run=dry_run)
            if not dry_run:
                await _drain_queue(settings, store, blink)
    finally:
        await blink.auth.session.close()


async def _run_probe(
    cameras: list[str],
    interval: float,
    duration: float | None,
    once: bool,
) -> None:
    settings = load_settings()
    try:
        blink = await connect(settings)
    except TwoFARequired as e:
        logging.error("%s", e)
        raise SystemExit(2)
    except AuthError as e:
        logging.error("%s", e)
        raise SystemExit(1)

    log = logging.getLogger("blink_watcher.probe")
    start_ts = datetime.now(timezone.utc)
    seen_ids: set[str] = set()
    tmp_root = Path(tempfile.mkdtemp(prefix="blink-probe-"))
    log.info("probe: cameras=%s interval=%.1fs tmp=%s start=%s", cameras, interval, tmp_root, start_ts.isoformat())

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal() -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal)
        except NotImplementedError:
            pass  # Windows

    deadline = None if duration is None else loop.time() + duration

    try:
        while not stop.is_set():
            try:
                async for clip in blink_api.list_clips(blink, cameras, since=start_ts):
                    if clip.id in seen_ids:
                        continue
                    seen_ids.add(clip.id)
                    dest = tmp_root / clip.camera / f"{clip.ts.strftime('%Y%m%d-%H%M%S')}-{clip.id}.mp4"
                    log.info("probe: new clip id=%s camera=%s ts=%s", clip.id, clip.camera, clip.ts.isoformat())
                    try:
                        await blink_api.download_clip(blink, clip, dest)
                        await asyncio.to_thread(process_video, dest)
                    except Exception as e:
                        log.exception("probe: failed to process clip %s: %s", clip.id, e)
                    finally:
                        try:
                            dest.unlink(missing_ok=True)
                        except Exception:
                            pass
            except Exception as e:
                log.exception("probe: list_clips failed: %s", e)

            if once:
                break
            if deadline is not None and loop.time() >= deadline:
                log.info("probe: duration elapsed")
                break
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
    finally:
        await blink.auth.session.close()
        shutil.rmtree(tmp_root, ignore_errors=True)


def cli() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="blink-watcher")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="Run live poller + downloader")

    t = sub.add_parser("telegram", help="Send a message to the configured Telegram chat")
    t.add_argument("text", help="Message text")
    t.add_argument("--markdown", action="store_true", help="Parse text as MarkdownV2")

    b = sub.add_parser("backfill", help="Backfill historical clips, newest-first")
    grp = b.add_mutually_exclusive_group()
    grp.add_argument("--hours", type=int, help="Hours of history (1..168)")
    grp.add_argument("--days", type=int, help="Days of history (1..7)")
    b.add_argument("--restart", action="store_true", help="Ignore saved cursor and start from now")
    b.add_argument("--dry-run", action="store_true", help="Count clips without enqueueing")
    b.add_argument("--then-live", action="store_true", help="Transition to live mode after backfill")

    p = sub.add_parser("probe", help="Poll given cameras and feed new clips directly to the event receiver")
    p.add_argument("--camera", action="append", required=True, help="Camera name (repeatable)")
    p.add_argument("--interval", type=float, default=5.0, help="Poll interval seconds (default: 5)")
    p.add_argument("--duration", type=float, default=None, help="Stop after N seconds (default: until Ctrl+C)")
    p.add_argument("--once", action="store_true", help="Poll once then exit")

    args = parser.parse_args()

    if args.cmd == "run":
        asyncio.run(_run_live())
    elif args.cmd == "backfill":
        if args.hours is not None:
            hours = args.hours
        elif args.days is not None:
            hours = args.days * 24
        else:
            hours = 24
        asyncio.run(_run_backfill(hours, args.restart, args.dry_run, args.then_live))
    elif args.cmd == "telegram":
        asyncio.run(_send_telegram(args.text, args.markdown))
    elif args.cmd == "probe":
        asyncio.run(_run_probe(args.camera, args.interval, args.duration, args.once))


if __name__ == "__main__":
    cli()
