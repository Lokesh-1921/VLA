"""Dataset-neutral sample schema and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np


@dataclass(slots=True)
class RobotSample:
    """Common representation; `image` may be an array or a path/URI string."""

    image: np.ndarray | str | None
    instruction: str
    action: np.ndarray
    episode_id: str
    timestep: int
    source: str
    state: np.ndarray | None = None
    embodiment: str | None = None
    next_image: np.ndarray | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = np.asarray(self.action, dtype=np.float32).reshape(-1)
        if not self.instruction.strip():
            raise ValueError("RobotSample instruction cannot be empty")
        if self.timestep < 0:
            raise ValueError("RobotSample timestep cannot be negative")
        if self.state is not None:
            self.state = np.asarray(self.state, dtype=np.float32).reshape(-1)

    def to_record(self) -> dict[str, Any]:
        return {
            "image": self.image if isinstance(self.image, str) or self.image is None else None,
            "instruction": self.instruction,
            "state": None if self.state is None else self.state.tolist(),
            "action": self.action.tolist(),
            "episode_id": self.episode_id,
            "timestep": self.timestep,
            "source": self.source,
            "embodiment": self.embodiment,
            "next_image": self.next_image if isinstance(self.next_image, str) or self.next_image is None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RobotSample":
        required = {"instruction", "action", "episode_id", "timestep", "source"}
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(f"Dataset record missing required fields: {', '.join(missing)}")
        return cls(
            image=record.get("image"),
            instruction=str(record["instruction"]),
            state=record.get("state"),
            action=np.asarray(record["action"], dtype=np.float32),
            episode_id=str(record["episode_id"]),
            timestep=int(record["timestep"]),
            source=str(record["source"]),
            embodiment=record.get("embodiment"),
            next_image=record.get("next_image"),
            metadata=dict(record.get("metadata", {})),
        )


class BaseRobotDataset(ABC, Sequence[RobotSample]):
    """Stable interface implemented by Bridge, RLDS, and custom adapters."""

    name: str = "base"

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> RobotSample: ...

    def __iter__(self) -> Iterator[RobotSample]:
        for index in range(len(self)):
            yield self[index]

    def action_dimension(self) -> int | None:
        return self[0].action.size if len(self) else None
