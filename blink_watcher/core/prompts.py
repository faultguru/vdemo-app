"""Prompt files: prompts/<name> at the project root."""
from __future__ import annotations

from blink_watcher.core.paths import PROMPTS_DIR


def load_prompt(name: str) -> str:
    """A prompt file without its # comment lines and surrounding whitespace."""
    lines = (PROMPTS_DIR / name).read_text(encoding="utf-8").splitlines()
    return "\n".join(ln for ln in lines if not ln.lstrip().startswith("#")).strip()
