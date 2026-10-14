#!/usr/bin/env python3
"""Inspect normalized JSONL robot records without loading all image pixels into memory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from visionpilot.datasets.base import RobotSample
from visionpilot.utils.logging import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="JSONL file or directory containing normalized JSONL files")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    files = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    if not files:
        parser.error(f"No JSONL records found under {root}; inspect a small extracted/normalized sample first.")
    episodes, actions, sources, embodiments, instructions = set(), [], Counter(), Counter(), Counter()
    image_refs, steps, invalid = 0, 0, []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    sample = RobotSample.from_record(json.loads(line))
                except Exception as exc:
                    invalid.append(f"{file}:{line_no}: {exc}")
                    continue
                episodes.add(sample.episode_id); actions.append(sample.action.size); sources[sample.source] += 1
                embodiments[sample.embodiment or "unknown"] += 1; instructions[sample.instruction] += 1; steps += 1
                image_refs += int(sample.image is not None)
    report = {
        "files": [str(file) for file in files], "episodes": len(episodes), "steps": steps, "image_references": image_refs,
        "action_dimensions": dict(Counter(actions)), "sources": dict(sources), "embodiments": dict(embodiments),
        "unique_instructions": len(instructions), "top_instructions": instructions.most_common(20),
        "invalid_records": invalid[:50], "approx_training_samples": steps,
        "note": "Image resolutions require image files to be present; this report deliberately does not decode every frame.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.write_report:
        destination = _bootstrap.ROOT / "data" / "metadata" / f"{root.stem}_inspection.json"
        write_json(destination, report)
        print(f"Wrote report: {destination}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
