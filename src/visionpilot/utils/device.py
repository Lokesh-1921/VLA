"""Portable hardware and optional-runtime reporting; imports no heavy ML package eagerly."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - available in core requirements
    psutil = None


def package_status(name: str) -> dict[str, Any]:
    installed = importlib.util.find_spec(name) is not None
    version = None
    if installed:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "importable (distribution version unavailable)"
    return {"installed": installed, "version": version}


def _torch_status() -> dict[str, Any]:
    status = package_status("torch")
    status["mps_available"] = False
    if not status["installed"]:
        return status
    try:
        import torch

        status["version"] = torch.__version__
        status["mps_available"] = bool(torch.backends.mps.is_available())
        status["mps_built"] = bool(torch.backends.mps.is_built())
        status["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # an incompatible wheel should be reported, not crash diagnostics
        status["import_error"] = f"{type(exc).__name__}: {exc}"
    return status


def _memory_bytes() -> int | None:
    if psutil is not None:
        return int(psutil.virtual_memory().total)
    return None


def system_report(path: str | Path = ".") -> dict[str, Any]:
    target = Path(path).resolve()
    disk = shutil.disk_usage(target)
    torch_status = _torch_status()
    mlx_status = package_status("mlx")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cpu_logical_count": psutil.cpu_count(logical=True) if psutil else None,
        "memory_bytes": _memory_bytes(),
        "disk": {"path": str(target), "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "torch": torch_status,
        "mlx": mlx_status,
        "transformers": package_status("transformers"),
        "peft": package_status("peft"),
        "opencv": package_status("cv2"),
        "recommended_backend": (
            "mlx_probe_only" if mlx_status["installed"] else "mps" if torch_status["mps_available"] else "cpu_smoke"
        ),
    }


def bytes_human(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"
