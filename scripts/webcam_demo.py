#!/usr/bin/env python3
"""Run VisionPilot webcam inference with simulated, rate-limited action display only."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from visionpilot.inference.pipeline import InferencePipeline
from visionpilot.inference.webcam import run_webcam_demo
from visionpilot.models.base import NumpySmokePolicy
from visionpilot.models.openvla_adapter import OpenVLAAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--backend", choices=("smoke", "openvla"), default="smoke")
    parser.add_argument("--model-id", default="openvla/openvla-7b")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--allow-remote-model", action="store_true")
    args = parser.parse_args()
    if args.backend == "smoke":
        model = NumpySmokePolicy()
    else:
        model = OpenVLAAdapter(args.model_id).load(local_files_only=not args.allow_remote_model)
    print("SAFETY: simulated executor only. Press Esc to stop; no hardware control is implemented.")
    run_webcam_demo(InferencePipeline(model), args.instruction, args.camera_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
