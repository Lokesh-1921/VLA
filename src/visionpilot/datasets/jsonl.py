from __future__ import annotations

import json
from pathlib import Path

from .base import BaseRobotDataset, RobotSample


class JsonlRobotDataset(BaseRobotDataset):
    """Small, inspectable normalized dataset implementation."""

    def __init__(self, path: str | Path, name: str = "jsonl") -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Normalized dataset missing: {self.path}")
        self.name = name
        self._records: list[dict] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        self._records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {self.path}:{line_no}") from exc

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> RobotSample:
        return RobotSample.from_record(self._records[index])
