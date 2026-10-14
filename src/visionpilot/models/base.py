"""Common VLA model API plus a dependency-free diagnostic policy."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from visionpilot.datasets.base import RobotSample


class BaseVLAModel(ABC):
    model_id: str
    action_dim: int

    @abstractmethod
    def predict(self, image: np.ndarray | str | None, instruction: str) -> np.ndarray: ...

    def predict_sample(self, sample: RobotSample) -> np.ndarray:
        return self.predict(sample.image, sample.instruction)


def load_image_array(image: np.ndarray | str | None, image_size: int) -> np.ndarray:
    if image is None:
        return np.zeros((image_size, image_size, 3), dtype=np.float32)
    if isinstance(image, str):
        with Image.open(Path(image)) as opened:
            rgb = opened.convert("RGB").resize((image_size, image_size))
            return np.asarray(rgb, dtype=np.float32) / 255.0
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 image, got shape {array.shape}")
    array = array[..., :3]
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if array.max(initial=0.0) > 1.0:
        array /= 255.0
    return array


@dataclass
class ActionNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def identity(cls, action_dim: int) -> "ActionNormalizer":
        return cls(np.zeros(action_dim, dtype=np.float32), np.ones(action_dim, dtype=np.float32))

    def normalize(self, action: np.ndarray) -> np.ndarray:
        return (np.asarray(action, dtype=np.float32) - self.mean) / np.maximum(self.std, 1e-6)

    def unnormalize(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32) * self.std + self.mean


class NumpySmokePolicy(BaseVLAModel):
    """Tiny low-rank action head used only to prove the local research pipeline.

    Vision statistics and deterministic token hashing form features. A frozen base head
    plus trainable A/B matrices is an intentionally transparent LoRA-like diagnostic.
    """

    model_id = "visionpilot/numpy-smoke-v1"

    def __init__(self, action_dim: int = 7, rank: int = 4, alpha: float = 8.0, image_size: int = 64, seed: int = 42) -> None:
        if action_dim <= 0 or rank <= 0:
            raise ValueError("action_dim and rank must be positive")
        self.action_dim, self.rank, self.alpha, self.image_size = action_dim, rank, alpha, image_size
        self.feature_dim = 18
        rng = np.random.default_rng(seed)
        self.base_weight = rng.normal(0.0, 0.08, (action_dim, self.feature_dim)).astype(np.float32)
        self.lora_a = np.zeros((action_dim, rank), dtype=np.float32)
        self.lora_b = rng.normal(0.0, 0.02, (rank, self.feature_dim)).astype(np.float32)
        self.bias = np.zeros(action_dim, dtype=np.float32)
        self.normalizer = ActionNormalizer.identity(action_dim)

    def features(self, image: np.ndarray | str | None, instruction: str) -> np.ndarray:
        pixels = load_image_array(image, self.image_size)
        channels = pixels.reshape(-1, 3)
        visual = np.concatenate([channels.mean(0), channels.std(0), channels.min(0), channels.max(0)]).astype(np.float32)
        language = np.zeros(6, dtype=np.float32)
        for token in instruction.lower().split():
            slot = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=2).digest(), "little") % language.size
            language[slot] += 1.0
        language /= max(1.0, float(len(instruction.split())))
        return np.concatenate([visual, language]).astype(np.float32)

    def effective_weight(self) -> np.ndarray:
        return self.base_weight + (self.alpha / self.rank) * (self.lora_a @ self.lora_b)

    def predict(self, image: np.ndarray | str | None, instruction: str) -> np.ndarray:
        prediction = self.effective_weight() @ self.features(image, instruction) + self.bias
        return self.normalizer.unnormalize(prediction).astype(np.float32)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        return features @ self.effective_weight().T + self.bias

    def state_dict(self) -> dict[str, np.ndarray | float | int]:
        return {
            "base_weight": self.base_weight,
            "lora_a": self.lora_a,
            "lora_b": self.lora_b,
            "bias": self.bias,
            "normalizer_mean": self.normalizer.mean,
            "normalizer_std": self.normalizer.std,
            "action_dim": self.action_dim,
            "rank": self.rank,
            "alpha": self.alpha,
            "image_size": self.image_size,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray]) -> "NumpySmokePolicy":
        model = cls(int(state["action_dim"]), int(state["rank"]), float(state["alpha"]), int(state["image_size"]))
        for name in ("base_weight", "lora_a", "lora_b", "bias"):
            setattr(model, name, np.asarray(state[name], dtype=np.float32))
        model.normalizer = ActionNormalizer(np.asarray(state["normalizer_mean"]), np.asarray(state["normalizer_std"]))
        return model

    def fit_normalizer(self, samples: Iterable[RobotSample]) -> None:
        actions = np.asarray([sample.action for sample in samples], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected actions with dimension {self.action_dim}")
        self.normalizer = ActionNormalizer(actions.mean(axis=0), np.maximum(actions.std(axis=0), 1e-3))
