#!/usr/bin/env python3
"""Evaluate a recorded prediction file; never invents task-success labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from visionpilot.evaluation.metrics import continuous_metrics, discrete_metrics
from visionpilot.evaluation.reports import write_markdown_report
from visionpilot.utils.config import load_config
from visionpilot.utils.logging import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True, help="JSONL with prediction and target fields")
    parser.add_argument("--output", default="experiments/evaluation")
    args = parser.parse_args()
    config, source = load_config(args.config), Path(args.predictions)
    if not source.is_file():
        parser.error(f"Predictions missing: {source}")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        parser.error("Prediction file is empty")
    action_space = config.get("evaluation", {}).get("action_space", "continuous")
    try:
        if action_space == "continuous":
            metrics = continuous_metrics([record["prediction"] for record in records], [record["target"] for record in records])
        elif action_space == "discrete":
            metrics = discrete_metrics([record["prediction"] for record in records], [record["target"] for record in records])
        else:
            parser.error(f"Unknown action space: {action_space}")
    except KeyError as exc:
        parser.error(f"Every prediction line needs {exc.args[0]!r}")
    output = Path(args.output)
    metadata = {"config": args.config, "predictions": str(source), "sample_count": len(records), "action_space": action_space}
    write_json(output / "metrics.json", {"metrics": metrics, "metadata": metadata})
    write_markdown_report(output / "report.md", "VisionPilot Evaluation", metrics, metadata)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
