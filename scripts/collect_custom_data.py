#!/usr/bin/env python3
"""Start explicit webcam collection of keyboard-labelled, simulated actions."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from visionpilot.data_collection.collector import WebcamCollector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    WebcamCollector(args.output, args.instruction, args.camera_index).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
