"""Native robot-action and local-demo action schemas are intentionally distinct."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DiscreteAction(str, Enum):
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    STOP = "STOP"
    LOOK_LEFT = "LOOK_LEFT"
    LOOK_RIGHT = "LOOK_RIGHT"
    LOOK_CENTER = "LOOK_CENTER"


@dataclass(frozen=True, slots=True)
class ContinuousAction:
    """Native VLA action vector with explicit provenance and optional normalization."""

    values: np.ndarray
    source: str = "model"
    normalized: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", np.asarray(self.values, dtype=np.float32).reshape(-1))
        if self.values.size == 0:
            raise ValueError("Continuous action cannot be empty")
        if not np.isfinite(self.values).all():
            raise ValueError("Continuous action contains non-finite values")
