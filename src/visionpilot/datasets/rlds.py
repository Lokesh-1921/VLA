"""RLDS adapter boundary. Keep TensorFlow/RLDS optional until a release is pinned."""

from __future__ import annotations

from .base import BaseRobotDataset


class RLDSDataset(BaseRobotDataset):
    name = "rlds"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "RLDS support must be configured against a pinned dataset builder and mapped "
            "to RobotSample. This prevents silently guessing action/image keys."
        )

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise IndexError(index)
