"""Adapters are presentation/control glue, never a replacement for native action labels."""

from __future__ import annotations

import numpy as np

from .schema import ContinuousAction, DiscreteAction


class ContinuousToDiscreteAdapter:
    """Map a conventional [x, y, z, roll, pitch, yaw, gripper] action to demo labels."""

    def __init__(self, deadband: float = 0.15) -> None:
        if deadband <= 0:
            raise ValueError("deadband must be positive")
        self.deadband = deadband

    def adapt(self, action: ContinuousAction | np.ndarray) -> DiscreteAction:
        values = action.values if isinstance(action, ContinuousAction) else np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size < 2:
            raise ValueError("Demo adapter needs at least x and y action components")
        x, y = float(values[0]), float(values[1])
        if max(abs(x), abs(y)) < self.deadband:
            if values.size >= 6 and abs(float(values[5])) >= self.deadband:
                return DiscreteAction.LOOK_LEFT if values[5] > 0 else DiscreteAction.LOOK_RIGHT
            return DiscreteAction.STOP
        if abs(x) >= abs(y):
            return DiscreteAction.MOVE_RIGHT if x > 0 else DiscreteAction.MOVE_LEFT
        return DiscreteAction.MOVE_FORWARD if y > 0 else DiscreteAction.MOVE_BACKWARD

    @staticmethod
    def keyboard_action(label: DiscreteAction) -> np.ndarray:
        vectors = {
            DiscreteAction.MOVE_LEFT: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            DiscreteAction.MOVE_RIGHT: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            DiscreteAction.MOVE_FORWARD: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            DiscreteAction.MOVE_BACKWARD: [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            DiscreteAction.LOOK_LEFT: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            DiscreteAction.LOOK_RIGHT: [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            DiscreteAction.LOOK_CENTER: [0.0] * 7,
            DiscreteAction.STOP: [0.0] * 7,
        }
        return np.asarray(vectors[label], dtype=np.float32)
