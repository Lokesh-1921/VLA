"""BridgeData V2 adapter after its upstream layout is inspected and normalized."""

from __future__ import annotations

from pathlib import Path

from .jsonl import JsonlRobotDataset


class BridgeV2Dataset(JsonlRobotDataset):
    name = "bridge_v2"

    def __init__(self, normalized_jsonl: str | Path) -> None:
        super().__init__(normalized_jsonl, name=self.name)
