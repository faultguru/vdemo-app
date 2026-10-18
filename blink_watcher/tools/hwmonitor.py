"""Background system monitor for GPU experiments - survives hard resets.

Every INTERVAL seconds writes one JSON line to STORAGE_ROOT/logs/hw_monitor.log (fsynced):
GPU 3D / video-encode / video-decode / compute utilization (summed over
processes, from Windows' GPU performance counters), dedicated + shared GPU
memory in use, CPU %, free RAM, and the top CPU processes. After a crash, the
tail of this file shows what the machine was doing in its last seconds;
compare its timestamps with hw_trace.log (blink_watcher.vision.hwtrace).

Usage (run in a separate terminal / background, stop with Ctrl+C):
    python -m blink_watcher hwmonitor
    python -m blink_watcher hwmonitor --interval 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from typing import Optional

import psutil

from blink_watcher.core.runtime import logs_dir
from blink_watcher.settings import get_settings
from blink_watcher.vision.hwtrace import durable_append

MONITOR_FILE = "hw_monitor.log"

# One persistent PowerShell that prints a JSON line of GPU counters per request.
_PS_LOOP = r"""
$ErrorActionPreference = 'SilentlyContinue'
$counters = @(
  '\GPU Engine(*)\Utilization Percentage',
  '\GPU Adapter Memory(*)\Dedicated Usage',
  '\GPU Adapter Memory(*)\Shared Usage'
)
while ($true) {
  $null = [Console]::In.ReadLine()
  $s = (Get-Counter -Counter $counters).CounterSamples
  $out = @{ gpu_3d = 0.0; gpu_compute = 0.0; gpu_encode = 0.0; gpu_decode = 0.0; gpu_copy = 0.0; ded_mb = 0.0; shared_mb = 0.0 }
  foreach ($c in $s) {
    $p = $c.Path
    if ($p -like '*utilization percentage') {
      if ($p -like '*engtype_3d*') { $out.gpu_3d += $c.CookedValue }
      elseif ($p -like '*engtype_compute*') { $out.gpu_compute += $c.CookedValue }
      elseif ($p -like '*engtype_videoencode*') { $out.gpu_encode += $c.CookedValue }
      elseif ($p -like '*engtype_videodecode*') { $out.gpu_decode += $c.CookedValue }
      elseif ($p -like '*engtype_copy*') { $out.gpu_copy += $c.CookedValue }
    }
    elseif ($p -like '*dedicated usage') { $out.ded_mb += $c.CookedValue / 1MB }
    elseif ($p -like '*shared usage') { $out.shared_mb += $c.CookedValue / 1MB }
  }
  foreach ($k in @($out.Keys)) { $out[$k] = [math]::Round($out[$k], 1) }
  [Console]::Out.WriteLine(($out | ConvertTo-Json -Compress))
  [Console]::Out.Flush()
}
"""


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=2.0, help="Seconds between samples (default 2)")
    args = ap.parse_args(argv)
    monitor_file = logs_dir(get_settings()) / MONITOR_FILE

    ps = subprocess.Popen(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_LOOP],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    psutil.cpu_percent()
    for p in psutil.process_iter():
        try:
            p.cpu_percent()
        except psutil.Error:
            pass
    durable_append(monitor_file, json.dumps({"ts": datetime.now().isoformat(timespec="milliseconds"), "event": "monitor.start"}))
    try:
        while True:
            t0 = time.monotonic()
            ps.stdin.write("\n")
            ps.stdin.flush()
            line = ps.stdout.readline()
            try:
                gpu = json.loads(line) if line.strip() else {}
            except json.JSONDecodeError:
                gpu = {"raw": line.strip()[:200]}
            top = []
            for p in psutil.process_iter(["name"]):
                try:
                    c = p.cpu_percent()
                except psutil.Error:
                    continue
                if c >= 5 and p.pid != 0:  # pid 0 = System Idle Process
                    top.append((c, p.info["name"], p.pid))
            top.sort(reverse=True)
            vm = psutil.virtual_memory()
            rec = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                **gpu,
                "cpu": psutil.cpu_percent(),
                "ram_free_mb": vm.available // 2**20,
                "top": [f"{n}:{pid}:{c:.0f}%" for c, n, pid in top[:5]],
            }
            durable_append(monitor_file, json.dumps(rec))
            time.sleep(max(0.0, args.interval - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        ps.kill()
        durable_append(monitor_file, json.dumps({"ts": datetime.now().isoformat(timespec="milliseconds"), "event": "monitor.stop"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
