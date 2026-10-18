"""Fixed locations inside the project: the source tree's own folders.

Runtime files (logs, models, state) live under STORAGE_ROOT — see
blink_watcher.core.runtime."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "blink_watcher"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
