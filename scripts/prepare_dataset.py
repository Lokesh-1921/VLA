#!/usr/bin/env python3
"""Validate normalized JSONL and write deterministic episode-level train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from visionpilot.datasets.base import RobotSample
from visionpilot.utils.logging import write_json


def split_for(episode_id: str, seed: int, train_fraction: float, val_fraction: float) -> str:
    value = int.from_bytes(hashlib.blake2b(f"{seed}:{episode_id}".encode(), digest_size=8).digest(), "big") / 2**64
    return "train" if value < train_fraction else "validation" if value < train_fraction + val_fraction else "test"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1 or not 0 <= args.validation_fraction < 1 or args.train_fraction + args.validation_fraction >= 1:
        parser.error("Split fractions must be positive and sum to less than 1")
    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.is_file():
        parser.error(f"Input not found: {input_path}")
    records, splits = [], {"train": [], "validation": [], "test": []}
    with input_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                sample = RobotSample.from_record(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid record at {input_path}:{number}") from exc
            record = sample.to_record()
            records.append(record)
            splits[split_for(sample.episode_id, args.seed, args.train_fraction, args.validation_fraction)].append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    for split, values in splits.items():
        split_path = output_path.with_name(f"{output_path.stem}_{split}.jsonl")
        with split_path.open("w", encoding="utf-8") as handle:
            for record in values:
                handle.write(json.dumps(record) + "\n")
    metadata = {"input": str(input_path), "output": str(output_path), "samples": len(records), "split_sizes": {key: len(value) for key, value in splits.items()}, "seed": args.seed}
    write_json(_bootstrap.ROOT / "data" / "metadata" / f"{output_path.stem}_preparation.json", metadata)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
