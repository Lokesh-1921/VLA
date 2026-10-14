from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from visionpilot.actions.adapter import ContinuousToDiscreteAdapter
from visionpilot.actions.schema import ContinuousAction, DiscreteAction
from visionpilot.models.base import BaseVLAModel


@dataclass(frozen=True)
class InferenceResult:
    action: ContinuousAction
    demo_action: DiscreteAction
    confidence: float
    latency_ms: float


class InferencePipeline:
    def __init__(self, model: BaseVLAModel, adapter: ContinuousToDiscreteAdapter | None = None) -> None:
        self.model = model
        self.adapter = adapter or ContinuousToDiscreteAdapter()

    def infer(self, image: np.ndarray | str | None, instruction: str) -> InferenceResult:
        started = time.perf_counter()
        values = self.model.predict(image, instruction)
        latency_ms = (time.perf_counter() - started) * 1000
        action = ContinuousAction(values)
        # Smoke confidence is a calibrated-display placeholder, not an uncertainty estimate.
        confidence = float(np.clip(1.0 / (1.0 + np.linalg.norm(values)), 0.0, 1.0))
        return InferenceResult(action, self.adapter.adapt(action), confidence, latency_ms)
