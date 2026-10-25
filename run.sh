#!/bin/sh
# Every service in one console: python -m blink_watcher run [--only ...] [--skip ...]
# Uses the project's .venv when there is one.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then py=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then py=.venv/Scripts/python.exe
elif python3 -c "" 2>/dev/null; then py=python3   # on Windows python3 may be the Store placeholder
else py=python
fi
exec "$py" -m blink_watcher run "$@"
