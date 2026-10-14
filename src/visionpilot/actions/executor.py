"""Safety-first action executors. Physical hardware must be added in a separate review."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .schema import DiscreteAction


class ActionExecutor(ABC):
    @abstractmethod
    def execute(self, action: DiscreteAction, confidence: float) -> bool: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...


@dataclass
class SimulatedExecutor(ActionExecutor):
    min_confidence: float = 0.65
    max_actions_per_second: float = 2.0
    history: list[dict[str, object]] = field(default_factory=list)
    _last_action_time: float = field(default=0.0, init=False)
    _stopped: bool = field(default=False, init=False)

    def execute(self, action: DiscreteAction, confidence: float) -> bool:
        now = time.monotonic()
        allowed = (
            not self._stopped
            and confidence >= self.min_confidence
            and now - self._last_action_time >= 1.0 / self.max_actions_per_second
        )
        self.history.append({"time": now, "action": action.value, "confidence": confidence, "accepted": allowed})
        if allowed:
            self._last_action_time = now
        return allowed

    def emergency_stop(self) -> None:
        self._stopped = True
        self.history.append({"time": time.monotonic(), "action": DiscreteAction.STOP.value, "confidence": 1.0, "accepted": True})


class KeyboardExecutor(SimulatedExecutor):
    """A dry-run executor for showing UI events; it has no OS-level key injection."""

    def execute(self, action: DiscreteAction, confidence: float) -> bool:
        return super().execute(action, confidence)
