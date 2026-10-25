@echo off
rem Every service in one console: python -m blink_watcher run [--only ...] [--skip ...]
rem Uses the project's .venv when there is one. Double-click to start; Ctrl+C stops.
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" -m blink_watcher run %*
pause
