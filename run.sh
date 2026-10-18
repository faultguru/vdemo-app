#!/bin/sh
# Every service in one console: python -m blink_watcher run [--only ...] [--skip ...]
# Uses the project's .venv when there is one.
cd "$(dirname "$0")" || exit 1
py=python3
[ -x .venv/bin/python ] && py=.venv/bin/python
exec "$py" -m blink_watcher run "$@"
