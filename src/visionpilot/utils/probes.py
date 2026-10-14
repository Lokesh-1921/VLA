"""Precise hardware and runtime probes with three-level reporting.

Each probe returns a dict distinguishing:
  - installed: the package imports
  - available: the hardware backend works
  - openvla_compatible: everything needed for OpenVLA is present

MLX installation never implies OpenVLA support.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    """Return installed version or None."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if importlib.util.find_spec(name) is not None:
            return "importable (version unavailable)"
        return None


def probe_pytorch_mps() -> dict[str, Any]:
    """Probe PyTorch installation and MPS backend availability."""
    version = _package_version("torch")
    result: dict[str, Any] = {
        "installed": version is not None,
        "version": version,
        "mps_built": False,
        "mps_available": False,
        "cuda_available": False,
        "openvla_compatible": False,
        "detail": "",
    }
    if not result["installed"]:
        result["detail"] = "PyTorch is not installed."
        return result
    try:
        import torch

        result["version"] = torch.__version__
        result["mps_built"] = bool(torch.backends.mps.is_built())
        result["mps_available"] = bool(torch.backends.mps.is_available())
        result["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        result["detail"] = f"PyTorch import error: {type(exc).__name__}: {exc}"
        return result

    if result["mps_available"]:
        result["detail"] = (
            "PyTorch MPS backend is available. OpenVLA compatibility "
            "depends on checkpoint, memory, and runtime — see probe_openvla_readiness()."
        )
    elif result["cuda_available"]:
        result["detail"] = "PyTorch CUDA backend is available."
    elif result["mps_built"]:
        result["detail"] = (
            "PyTorch was built with MPS support but MPS is not available on this system."
        )
    else:
        result["detail"] = "PyTorch is installed but no GPU backend (MPS/CUDA) is available."
    return result


def probe_mlx() -> dict[str, Any]:
    """Probe MLX installation. Never claims OpenVLA compatibility."""
    version = _package_version("mlx")
    result: dict[str, Any] = {
        "installed": version is not None,
        "version": version,
        "openvla_compatible": False,
        "detail": "",
    }
    if not result["installed"]:
        result["detail"] = "MLX is not installed."
    else:
        result["detail"] = (
            "MLX is installed, but no verified OpenVLA-compatible MLX "
            "conversion or runtime is configured in this project. "
            "Do not treat MLX availability as OpenVLA support."
        )
    return result


# Minimum memory thresholds (bytes) — conservative estimates.
_OPENVLA_INFERENCE_MIN_BYTES = 20 * 1024**3   # ~20 GiB
_OPENVLA_TRAINING_MIN_BYTES = 48 * 1024**3     # ~48 GiB


def probe_openvla_readiness(
    model_id: str = "openvla/openvla-7b",
    device: str | None = None,
    memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Aggregated preflight report for OpenVLA.

    Checks dependencies, device, memory, and checkpoint presence.
    Returns a structured report with a ``verdict`` field.
    """
    # --- dependency checks ---
    dep_names = ["torch", "transformers", "peft", "accelerate", "safetensors"]
    dependencies: dict[str, dict[str, Any]] = {}
    all_deps_ok = True
    for name in dep_names:
        ver = _package_version(name)
        dependencies[name] = {"installed": ver is not None, "version": ver}
        if ver is None:
            all_deps_ok = False

    # --- device check ---
    torch_probe = probe_pytorch_mps()
    if device is None:
        if torch_probe.get("mps_available"):
            device = "mps"
        elif torch_probe.get("cuda_available"):
            device = "cuda"
        else:
            device = "cpu"
    device_ok = device in ("mps", "cuda")

    # --- memory check ---
    if memory_bytes is None:
        try:
            import psutil
            memory_bytes = int(psutil.virtual_memory().total)
        except Exception:
            memory_bytes = 0
    inference_ok = memory_bytes >= _OPENVLA_INFERENCE_MIN_BYTES
    training_ok = memory_bytes >= _OPENVLA_TRAINING_MIN_BYTES

    # --- checkpoint check ---
    checkpoint_found = False
    checkpoint_detail = "Not checked (dependencies missing)."
    if all_deps_ok:
        try:
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(model_id, "config.json")
            checkpoint_found = result is not None and isinstance(result, str)
            checkpoint_detail = (
                f"Found in local cache: {result}" if checkpoint_found
                else "Not found in local HuggingFace cache."
            )
        except Exception:
            # huggingface_hub may not be installed or may have changed API
            try:
                from transformers.utils import cached_file
                cached = cached_file(model_id, "config.json", local_files_only=True, _raise_exceptions_for_missing_entries=False)
                checkpoint_found = cached is not None
                checkpoint_detail = (
                    f"Found in local cache: {cached}" if checkpoint_found
                    else "Not found in local Transformers cache."
                )
            except Exception:
                checkpoint_detail = "Could not probe local cache."

    # --- verdict ---
    issues: list[str] = []
    if not all_deps_ok:
        missing = [n for n, d in dependencies.items() if not d["installed"]]
        issues.append(f"missing_dependencies: {', '.join(missing)}")
    if not device_ok:
        issues.append(f"unsupported_device: {device} (need mps or cuda)")
    if not inference_ok:
        issues.append(
            f"insufficient_memory: {memory_bytes / 1024**3:.1f} GiB "
            f"(need ≥{_OPENVLA_INFERENCE_MIN_BYTES / 1024**3:.0f} GiB for inference)"
        )
    if not checkpoint_found:
        issues.append("missing_checkpoint")

    if not issues:
        verdict = "ready"
    elif not all_deps_ok:
        verdict = "missing_deps"
    elif not checkpoint_found:
        verdict = "missing_checkpoint"
    elif not inference_ok:
        verdict = "insufficient_memory"
    elif not device_ok:
        verdict = "unsupported_device"
    else:
        verdict = "not_ready"

    return {
        "model_id": model_id,
        "dependencies": dependencies,
        "device": device,
        "device_supported": device_ok,
        "memory_bytes": memory_bytes,
        "memory_sufficient_inference": inference_ok,
        "memory_sufficient_training": training_ok,
        "checkpoint_found": checkpoint_found,
        "checkpoint_detail": checkpoint_detail,
        "verdict": verdict,
        "issues": issues,
    }
