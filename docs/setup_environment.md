# Environment Setup for Apple Silicon

## Recommended Python version

Use **Python 3.12** for the best wheel compatibility with PyTorch, MLX, and
Transformers on Apple Silicon (arm64). Python 3.13 may work but has known gaps
for some ML packages. Python 3.11 is also safe.

Check which Pythons are available:

```bash
python3.12 --version   # preferred
python3.11 --version   # fallback
python3 --version       # system default — may be 3.13+
```

If `python3.12` is not installed, use Homebrew:

```bash
brew install python@3.12
```

## Creating the virtual environment

Always use a dedicated venv so ML wheels do not interfere with your system
Python or Conda environments.

```bash
cd /path/to/VLA

# Create and activate
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install -U pip

# Install VisionPilot core (numpy, Pillow, PyYAML, psutil)
python -m pip install -e .
```

## Verifying the base install

```bash
python scripts/check_system.py
python scripts/benchmark_model.py --backend smoke --iterations 20
python scripts/train.py --config configs/mac_mlx.yaml --smoke
PYTHONPATH=src python -m unittest discover -s tests -v
```

All four commands must succeed before installing optional extras.

## Optional extras

Install only what you need. Each extra is independent.

### Webcam UI (OpenCV)

```bash
python -m pip install -e '.[webcam]'
```

### PyTorch + OpenVLA runtime

```bash
python -m pip install -e '.[openvla]'
```

This installs PyTorch, Transformers, Accelerate, PEFT, and Safetensors.
After installing, verify MPS availability:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

> **Important**: Installing the OpenVLA extras makes the runtime *importable*,
> but does NOT mean OpenVLA-7B will run successfully. The full model requires:
>
> - A compatible OpenVLA checkpoint (local or downloaded)
> - Sufficient memory (~28+ GiB for inference, ~48+ GiB for training)
> - A verified `predict_action` API in the loaded model
>
> Run `python -c "from visionpilot.utils.probes import probe_openvla_readiness; import json; print(json.dumps(probe_openvla_readiness(), indent=2))"` for a full compatibility report.

### MLX (Apple Silicon native runtime)

```bash
python -m pip install -e '.[mlx]'
```

> **Warning**: Installing MLX does **not** imply OpenVLA support. There is
> currently no verified OpenVLA-compatible MLX conversion or runtime in this
> project. The MLX probe reports installation status only; it explicitly sets
> `openvla_compatible: false` until a conversion is validated.

## What the probes report

After installing any extras, the probe module provides three-level status:

| Level | Meaning |
|---|---|
| **installed** | The Python package imports successfully |
| **available** | The hardware backend (MPS/CUDA/MLX) is functional |
| **openvla_compatible** | All dependencies, checkpoint, device, and memory are sufficient for OpenVLA |

Use `scripts/check_system.py` for a quick overview, or import
`visionpilot.utils.probes` for programmatic access.
