"""Describe a frames folder (written by `python -m blink_watcher frames`) with
the LLM and save the answer as describe.result.json in that folder, next to
the request files of the call.

Usage:
    python -m blink_watcher describe path/to/clip.details
    python -m blink_watcher describe path/to/clip.details --model claude-haiku-4-5 --effort low
    python -m blink_watcher describe path/to/clip.details --provider claude-api
    python -m blink_watcher describe path/to/clip.details --dry-run   # write the request, don't send it
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, get_args

from blink_watcher.ai import LLMRequest, make_llm
from blink_watcher.core.logs import setup_logging
from blink_watcher.describe import RESULT_FILE
from blink_watcher.describe.describer import VideoDescriber
from blink_watcher.describe.request import build_request
from blink_watcher.describe.schema import VideoAnalysis
from blink_watcher.settings import load_settings
from blink_watcher.settings.ai import Effort, LLMProvider
from blink_watcher.vision.sampling import MANIFEST

log = logging.getLogger(__name__)

PROGRAM = "describe"


def main(argv: Optional[list[str]] = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames_dir", type=Path, help="A frames folder (manifest.json + JPEGs)")
    parser.add_argument("--model", default=settings.describer_model,
                        help=f"Model (default DESCRIBER_MODEL: {settings.describer_model})")
    parser.add_argument("--effort", default=settings.describer_effort or None,
                        choices=[e for e in get_args(Effort) if e],
                        help="Thinking effort (default DESCRIBER_EFFORT, else the model's default)")
    parser.add_argument("--provider", choices=get_args(LLMProvider), default=settings.llm_provider,
                        help=f"LLM provider (default LLM_PROVIDER: {settings.llm_provider})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the request files and print the user prompt; do not call the model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also show other modules' INFO logs")
    args = parser.parse_args(argv)
    setup_logging(PROGRAM, args.verbose, "blink_watcher.describe")

    if not (args.frames_dir / MANIFEST).exists():
        log.error("No %s in %s", MANIFEST, args.frames_dir)
        return 1
    llm = make_llm(settings.model_copy(update={"llm_provider": args.provider}))
    if args.dry_run:
        req = build_request(args.frames_dir)
        files = llm.write_request(LLMRequest(req.system, req.parts, VideoAnalysis, args.model, args.effort),
                                  args.frames_dir)
        log.info("Dry run: wrote %s", ", ".join(files))
        sys.stdout.reconfigure(encoding="utf-8")  # the prompt has non-ASCII; Windows consoles default to cp1252
        print(req.user_prompt)
        return 0

    description = VideoDescriber(llm, args.model, args.effort).describe_frames(args.frames_dir)
    if description.analysis is None:
        log.error("No structured answer (stop_reason=%s) - see raw_text in %s",
                  description.response.stop_reason, args.frames_dir / RESULT_FILE)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
