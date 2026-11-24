#!/usr/bin/env bash
# Read-only hardware and filesystem check for one assigned 910A host.
set -euo pipefail
runtime=/home/ma-user/work/CADRetrieval_runtime
python="$runtime/envs/cadretrieval-910a-cann80-torch240/bin/python"
"$python" -B - "$runtime" <<'PY'
from datetime import datetime, timezone
import json
import os
import re
import subprocess
import sys

runtime = sys.argv[1]
result = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, timeout=20, check=True)
devices = [
    {"physical_index": int(index), "health": status}
    for index, status in re.findall(
        r"^\|\s+(\d+)\s+910ProA\s+\|\s+(\S+)", result.stdout, re.MULTILINE
    )
]
space = os.statvfs(runtime)
available_bytes = space.f_bavail * space.f_frsize
issues = []
if sorted(device["physical_index"] for device in devices) != list(range(8)):
    issues.append("incomplete_device_inventory")
if any(device["health"] != "OK" for device in devices):
    issues.append("device_health")
if available_bytes < 5 * 1024**3:
    issues.append("runtime_volume_below_5_GiB")
if space.f_files and space.f_favail < 1000:
    issues.append("runtime_volume_low_inodes")
print(json.dumps({
    "checked_utc": datetime.now(timezone.utc).isoformat(),
    "runtime": runtime,
    "devices": devices,
    "available_bytes": available_bytes,
    "available_inodes": space.f_favail if space.f_files else None,
    "healthy": not issues,
    "issues": issues,
}, allow_nan=False))
raise SystemExit(2 if issues else 0)
PY
