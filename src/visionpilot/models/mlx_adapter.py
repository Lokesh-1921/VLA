"""MLX capability probe; no unverified OpenVLA conversion is represented as runnable."""

from __future__ import annotations

import importlib.util

from .base import BaseVLAModel


class MlxOpenVLAAdapter(BaseVLAModel):
    model_id = "openvla/openvla-7b"
    action_dim = 7

    @staticmethod
    def availability() -> tuple[bool, str]:
        if importlib.util.find_spec("mlx") is None:
            return False, "MLX is not installed"
        return False, "MLX is installed, but this project has no verified OpenVLA-compatible MLX conversion/runtime configured."

    def predict(self, image, instruction):
        _, reason = self.availability()
        raise RuntimeError(reason)
