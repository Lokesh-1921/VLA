#!/usr/bin/env python3
"""Report the preflight facts required before a large model/data operation."""

from __future__ import annotations

import argparse
import json
import subprocess

import _bootstrap  # noqa: F401
from visionpilot.utils.device import bytes_human, system_report


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit structured JSON")
    args = parser.parse_args()
    report = system_report(_bootstrap.ROOT)
    report["cpu_brand"] = _sysctl("machdep.cpu.brand_string") or report["processor"]
    physical_memory = _sysctl("hw.memsize")
    if physical_memory and physical_memory.isdigit():
        report["memory_bytes"] = int(physical_memory)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    torch = report["torch"]
    print("VisionPilot system preflight")
    print(f"DEVICE: {report['machine']} ({report['platform']})")
    print(f"CPU: {report['cpu_brand']}")
    print(f"RAM / unified memory: {bytes_human(report['memory_bytes'])}")
    print(f"GPU backend: {report['recommended_backend']}")
    print(f"MPS available: {torch['mps_available']}")
    print(f"MLX available: {report['mlx']['installed']}")
    print(f"Disk free at project: {bytes_human(report['disk']['free_bytes'])} / {bytes_human(report['disk']['total_bytes'])}")
    print(f"Python: {report['python']} ({report['executable']})")
    print(f"PyTorch: {torch['version'] or 'not installed'}")
    print(f"Transformers: {report['transformers']['version'] or 'not installed'}")
    print(f"PEFT: {report['peft']['version'] or 'not installed'}")
    if report["machine"] == "arm64" and not report["mlx"]["installed"] and not torch["mps_available"]:
        print("RESULT: Apple Silicon detected, but neither MLX nor an MPS-capable PyTorch build is installed.")
    elif report["machine"] == "arm64":
        print("RESULT: Apple Silicon path is detectable; validate a model-specific memory benchmark before training.")
    else:
        print("RESULT: Non-Apple-Silicon environment; use the CPU smoke path or a supported accelerator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
