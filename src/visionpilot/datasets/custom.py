from __future__ import annotations

from pathlib import Path

from .jsonl import JsonlRobotDataset


class CustomVisionPilotDataset(JsonlRobotDataset):
    name = "visionpilot_custom"

    def __init__(self, normalized_jsonl: str | Path) -> None:
        super().__init__(normalized_jsonl, name=self.name)
