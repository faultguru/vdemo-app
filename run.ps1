# Every service in one console: python -m blink_watcher run [--only ...] [--skip ...]
# Uses the project's .venv when there is one.
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Set-Location $PSScriptRoot
& $py -m blink_watcher run @args
exit $LASTEXITCODE
