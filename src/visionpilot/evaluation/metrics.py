from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def continuous_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    prediction_array = np.asarray(predictions, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    if prediction_array.shape != target_array.shape or prediction_array.size == 0:
        raise ValueError("Continuous predictions and targets must be non-empty and have equal shapes")
    residual = prediction_array - target_array
    return {"action_mae": float(np.mean(np.abs(residual))), "action_rmse": float(np.sqrt(np.mean(residual**2)))}


def discrete_metrics(predictions: Sequence[str], targets: Sequence[str]) -> dict[str, object]:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Discrete predictions and targets must be non-empty and have equal length")
    labels = sorted(set(predictions).union(targets))
    confusion: dict[str, dict[str, int]] = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for prediction, truth in zip(predictions, targets, strict=True):
        confusion[truth][prediction] += 1
    precision_values, recall_values, f1_values = [], [], []
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        false_negative = sum(confusion[label][other] for other in labels if other != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return {
        "accuracy": float(sum(p == t for p, t in zip(predictions, targets, strict=True)) / len(targets)),
        "precision_macro": float(np.mean(precision_values)),
        "recall_macro": float(np.mean(recall_values)),
        "f1_macro": float(np.mean(f1_values)),
        "confusion_matrix": confusion,
        "labels": labels,
        "support": dict(Counter(targets)),
    }
