"""LoRA configuration shared by experiment manifests and model-specific trainers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
