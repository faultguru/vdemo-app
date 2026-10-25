@echo off
rem vdemo: start / stop / restart the services in the background, or run them in this console.
rem Uses the project's .venv when there is one. "vdemo help" shows the usage.
setlocal
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

set "CMD=%~1"
if "%CMD%"=="" goto usage
if /i "%CMD%"=="help" goto usage
if /i "%CMD%"=="-h" goto usage
if /i "%CMD%"=="--help" goto usage
if /i "%CMD%"=="/?" goto usage
if /i "%CMD%"=="console" goto console
for %%c in (start stop restart status) do if /i "%CMD%"=="%%c" goto control
echo vdemo: unknown command "%CMD%" 1>&2
echo. 1>&2
call :usage 1>&2
exit /b 2

:control
"%PY%" -m blink_watcher control %CMD%
exit /b %ERRORLEVEL%

:console
shift
set "ARGS="
:collect
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto collect
:run
"%PY%" -m blink_watcher run%ARGS%
exit /b %ERRORLEVEL%

:usage
echo usage: vdemo ^<command^> [options]
echo.
echo commands:
echo   start     start every service in the background (no window) and return once the
echo             web console answers; already running: nothing happens
echo   stop      stop the services (the Ctrl+C way; killed after CONTROL_STOP_WAIT_SEC),
echo             also one running in a console window
echo   restart   stop, then start
echo   status    running or not (exit code 0 = running, 3 = not running)
echo   console   run every service in this console, output shown; Ctrl+C stops
echo             options: --only NAMES, --skip NAMES (fetch, process, notify, chat-bot,
echo             webui), -v (also other modules' INFO logs)
echo   help      this text
echo.
echo Only one instance runs at a time. Logs: STORAGE_ROOT\logs (run.log and one per service).
echo Health: http://127.0.0.1:9090/api/health (WEBUI_HOST / WEBUI_PORT).
exit /b 0
