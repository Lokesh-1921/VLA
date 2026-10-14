from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visionpilot.models.base import NumpySmokePolicy


def save_smoke_checkpoint(model: NumpySmokePolicy, path: str | Path, step: int) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    arrays = {key: value for key, value in state.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in state.items() if not isinstance(value, np.ndarray)}
    metadata["step"] = int(step)
    np.savez_compressed(destination, **arrays, metadata=json.dumps(metadata))
    return destination


def load_smoke_checkpoint(path: str | Path) -> tuple[NumpySmokePolicy, int]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        state = {name: payload[name] for name in payload.files if name != "metadata"}
    state.update({key: metadata[key] for key in ("action_dim", "rank", "alpha", "image_size")})
    return NumpySmokePolicy.from_state_dict(state), int(metadata["step"])
