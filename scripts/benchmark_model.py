#!/usr/bin/env python3
"""Measure the safe smoke policy or explicitly probe an already-local OpenVLA model."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import psutil

import _bootstrap  # noqa: F401
from visionpilot.models.base import NumpySmokePolicy, load_image_array
from visionpilot.models.openvla_adapter import OpenVLAAdapter


def _rss() -> int:
    return psutil.Process().memory_info().rss


def smoke_benchmark(iterations: int) -> dict[str, float | str]:
    before = _rss()
    loading_start = time.perf_counter()
    model = NumpySmokePolicy()
    loading_ms = (time.perf_counter() - loading_start) * 1000
    after_load = _rss()
    image = np.random.default_rng(42).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    preprocessing_start = time.perf_counter()
    for _ in range(iterations):
        load_image_array(image, 64)
    preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000 / iterations
    for _ in range(3):
        model.predict(image, "Move toward the bottle")
    infer_start = time.perf_counter()
    for _ in range(iterations):
        action = model.predict(image, "Move toward the bottle")
    infer_ms = (time.perf_counter() - infer_start) * 1000 / iterations
    return {
        "backend": "numpy_smoke (diagnostic; not OpenVLA)", "model_loading_ms": loading_ms,
        "process_memory_delta_bytes": _rss() - before, "model_load_memory_delta_bytes": after_load - before,
        "image_preprocessing_ms": preprocessing_ms, "inference_latency_ms": infer_ms,
        "actions_per_second": 1000.0 / infer_ms if infer_ms else 0.0, "action_dim": int(action.size),
    }


def openvla_probe(model_id: str, allow_remote_model: bool) -> dict[str, object]:
    available, detail = OpenVLAAdapter.availability()
    result: dict[str, object] = {"backend": "openvla", "runtime_available": available, "detail": detail, "remote_download_allowed": allow_remote_model}
    if not available:
        return result
    started, before = time.perf_counter(), _rss()
    try:
        adapter = OpenVLAAdapter(model_id=model_id)
        adapter.load(local_files_only=not allow_remote_model)
        result.update({"loaded": True, "model_loading_ms": (time.perf_counter() - started) * 1000, "memory_delta_bytes": _rss() - before})
    except Exception as exc:
        result.update({"loaded": False, "reason": f"{type(exc).__name__}: {exc}"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("smoke", "openvla"), default="smoke")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--model-id", default="openvla/openvla-7b")
    parser.add_argument("--allow-remote-model", action="store_true", help="allow a model hub download; default probes local cache only")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    result = smoke_benchmark(args.iterations) if args.backend == "smoke" else openvla_probe(args.model_id, args.allow_remote_model)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.backend == "smoke" or result.get("loaded") else 2


if __name__ == "__main__":
    raise SystemExit(main())
