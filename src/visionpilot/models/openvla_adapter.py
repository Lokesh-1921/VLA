"""Optional official-runtime boundary for a pretrained OpenVLA checkpoint."""

from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import Any

import numpy as np

from .base import BaseVLAModel


# Minimum memory thresholds (bytes) — conservative estimates for OpenVLA-7B.
_OPENVLA_INFERENCE_MIN_BYTES = 20 * 1024**3   # ~20 GiB
_OPENVLA_TRAINING_MIN_BYTES = 48 * 1024**3     # ~48 GiB


class OpenVLAAdapter(BaseVLAModel):
    """Load only from a user-provided compatible OpenVLA/Transformers installation.

    Exact loader APIs differ by OpenVLA release. This adapter verifies installed
    dependencies and uses the documented ``predict_action`` integration surface when
    available; it refuses a generic ``generate`` fallback because that could silently
    turn token text into an invalid robot action.

    Safety guards
    -------------
    - ``load()`` raises ``RuntimeError`` if the loaded model lacks ``predict_action``.
    - ``predict()`` raises ``RuntimeError`` if the model is not loaded.
    - ``preflight_report()`` returns a structured compatibility check without
      loading any model weights.
    """

    def __init__(self, model_id: str = "openvla/openvla-7b", device: str = "mps") -> None:
        self.model_id = model_id
        self.device = device
        self.model: Any | None = None
        self.processor: Any | None = None
        self.action_dim = 7

    @staticmethod
    def availability() -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return False, f"Missing optional runtime: {exc.name}"
        return True, "PyTorch and Transformers import successfully; official OpenVLA runtime still must be compatible."

    def load(self, *, local_files_only: bool = True) -> "OpenVLAAdapter":
        available, reason = self.availability()
        if not available:
            raise RuntimeError(reason)
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, local_files_only=local_files_only)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=dtype, local_files_only=local_files_only
            ).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "Could not load an OpenVLA-compatible checkpoint. Install/pin the official OpenVLA runtime and checkpoint first; "
                "do not interpret this as an MPS compatibility result."
            ) from exc
        if not hasattr(self.model, "predict_action"):
            raise RuntimeError("Loaded model has no predict_action API; refuse unsafe generic generation as a robot action.")
        return self

    def predict(self, image: np.ndarray | str | None, instruction: str) -> np.ndarray:
        if self.model is None or self.processor is None:
            raise RuntimeError("OpenVLA model is not loaded")
        if image is None:
            raise ValueError("OpenVLA prediction requires an image")
        from PIL import Image

        if isinstance(image, str):
            with Image.open(image) as opened:
                image_input = opened.convert("RGB")
        else:
            image_input = Image.fromarray(np.asarray(image, dtype=np.uint8))
        try:
            action = self.model.predict_action(self.processor, image_input, instruction, unnorm_key=None)
        except TypeError:
            action = self.model.predict_action(image_input, instruction)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    @classmethod
    def preflight_report(
        cls,
        model_id: str = "openvla/openvla-7b",
        device: str | None = None,
        memory_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Structured compatibility report without loading model weights.

        Returns a dict with keys: model_id, dependencies, device, device_supported,
        memory_bytes, memory_sufficient_inference, memory_sufficient_training,
        checkpoint_found, checkpoint_detail, verdict, issues.
        """

        def _pkg_ver(name: str) -> str | None:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                if importlib.util.find_spec(name) is not None:
                    return "importable (version unavailable)"
                return None

        # --- dependency checks ---
        dep_names = ["torch", "transformers", "peft", "accelerate", "safetensors"]
        dependencies: dict[str, dict[str, Any]] = {}
        all_deps_ok = True
        for name in dep_names:
            ver = _pkg_ver(name)
            dependencies[name] = {"installed": ver is not None, "version": ver}
            if ver is None:
                all_deps_ok = False

        # --- device check ---
        mps_ok = cuda_ok = False
        if all_deps_ok:
            try:
                import torch
                mps_ok = bool(torch.backends.mps.is_available())
                cuda_ok = bool(torch.cuda.is_available())
            except Exception:
                pass
        if device is None:
            device = "mps" if mps_ok else "cuda" if cuda_ok else "cpu"
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
                try:
                    from transformers.utils import cached_file
                    cached = cached_file(
                        model_id, "config.json",
                        local_files_only=True,
                        _raise_exceptions_for_missing_entries=False,
                    )
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
