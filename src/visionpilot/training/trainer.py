"""Actual tiny backpropagation path, deliberately separate from the future OpenVLA trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from visionpilot.datasets.base import RobotSample
from visionpilot.models.base import NumpySmokePolicy


@dataclass
class TrainResult:
    steps: int
    losses: list[float]

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


class SmokeTrainer:
    def __init__(self, model: NumpySmokePolicy, learning_rate: float = 1e-2) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.model, self.learning_rate = model, learning_rate

    def train(
        self,
        samples: Sequence[RobotSample],
        epochs: int = 1,
        micro_batch_size: int = 1,
        checkpoint_callback: Callable[[int], None] | None = None,
    ) -> TrainResult:
        if not samples:
            raise ValueError("Cannot train with zero samples")
        if micro_batch_size <= 0 or epochs <= 0:
            raise ValueError("epochs and micro_batch_size must be positive")
        self.model.fit_normalizer(samples)
        features = np.vstack([self.model.features(s.image, s.instruction) for s in samples])
        targets = np.vstack([self.model.normalizer.normalize(s.action) for s in samples])
        losses: list[float] = []
        step = 0
        for _ in range(epochs):
            for start in range(0, len(samples), micro_batch_size):
                x = features[start : start + micro_batch_size]
                y = targets[start : start + micro_batch_size]
                prediction = self.model.predict_features(x)
                residual = prediction - y
                loss = float(np.mean(residual**2))
                gradient_weight = (2.0 / residual.size) * residual.T @ x
                scale = self.model.alpha / self.model.rank
                gradient_a = scale * gradient_weight @ self.model.lora_b.T
                gradient_b = scale * self.model.lora_a.T @ gradient_weight
                gradient_bias = (2.0 / residual.size) * residual.sum(axis=0)
                self.model.lora_a -= self.learning_rate * gradient_a.astype(np.float32)
                self.model.lora_b -= self.learning_rate * gradient_b.astype(np.float32)
                self.model.bias -= self.learning_rate * gradient_bias.astype(np.float32)
                step += 1
                losses.append(loss)
                if checkpoint_callback:
                    checkpoint_callback(step)
        return TrainResult(step, losses)
