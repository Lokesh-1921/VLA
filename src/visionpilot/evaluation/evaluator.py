from __future__ import annotations

from typing import Iterable

import numpy as np

from visionpilot.datasets.base import RobotSample
from visionpilot.models.base import BaseVLAModel

from .metrics import continuous_metrics


def evaluate_predictions(model: BaseVLAModel, samples: Iterable[RobotSample]) -> dict[str, float]:
    items = list(samples)
    if not items:
        raise ValueError("Evaluation requires samples")
    predicted = np.vstack([model.predict_sample(sample) for sample in items])
    targets = np.vstack([sample.action for sample in items])
    return continuous_metrics(predicted, targets)
