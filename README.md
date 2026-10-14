# VisionPilot

VisionPilot is a reproducible research scaffold for studying **vision-language-action (VLA) domain adaptation**:

`image + instruction -> robot action`

It intentionally does not claim a new foundation model. The research story is: evaluate a pretrained OpenVLA baseline, adapt it with BridgeData V2 using parameter-efficient fine-tuning, adapt again on a small local dataset, then compare all three models with one fixed protocol.

## Status and safety

This first milestone is deliberately local-first and stops before any BridgeData download. It includes hardware checks, a dependency-free diagnostic policy that validates the complete data-to-loss-to-checkpoint path, an optional OpenVLA integration boundary, dataset inspection/preparation tools, and a webcam UI with simulated actions only. Hardware execution is disabled by default; no model is allowed to issue physical commands from this repository.

The diagnostic `numpy_smoke` model is **not OpenVLA** and must never be reported as an OpenVLA result. It exists so a Mac can prove the project wiring before downloading 124 GB or loading a 7B checkpoint.

## Architecture

```text
camera / dataset frame + instruction
              |
       normalized RobotSample
              |
  OpenVLA adapter (optional) or smoke policy
              |
       continuous model action
              |
       action adapter -> demo label
              |
       simulated, rate-limited executor
```

The normalized sample retains frame, instruction, state, native action, episode/timestep, source, and embodiment metadata. Dataset implementations remain behind `BaseRobotDataset`, so BridgeData, RLDS, and local collections can be mixed later without putting dataset-specific assumptions in training code.

## Quick start

Use a dedicated Python 3.12 virtual environment for PyTorch/MLX wheel compatibility. See [`docs/setup_environment.md`](docs/setup_environment.md) for detailed setup instructions.

```bash
python3 scripts/check_system.py
python3 scripts/benchmark_model.py --backend smoke --iterations 20
python3 scripts/train.py --config configs/mac_mlx.yaml --smoke
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The smoke training command creates a uniquely named experiment under `experiments/`, writes metrics/configuration, and validates real gradient updates and checkpoint restore on synthetic normalized samples. It is the required gate before a dataset download or an expensive run.

## Installing optional research dependencies

Install only after the system report identifies a supported Python. See [`docs/setup_environment.md`](docs/setup_environment.md) for the recommended venv creation steps.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
# Optional extras (install only what you need):
python -m pip install -e '.[openvla]'   # PyTorch + Transformers + PEFT
python -m pip install -e '.[mlx]'       # Apple Silicon MLX (does NOT imply OpenVLA support)
python -m pip install -e '.[webcam]'    # OpenCV webcam UI
```

`requirements.txt` lists optional training integrations instead of pretending that OpenVLA-7B is viable in 16 GB unified memory. The official OpenVLA repository and checkpoint must be installed/configured separately; set `model.id` in a run config and invoke `--backend openvla` for a load-only compatibility probe. OpenVLA fine-tuning is normally a CUDA/Linux workflow; MLX compatibility for the full OpenVLA architecture must be verified against a maintained conversion/runtime before it is treated as supported.

## Data workflow (no automatic download)

### Mode 1: TFDS directory (official BridgeData V2 release)

The official OpenVLA documentation describes BridgeData V2 as a recursively downloaded TFDS directory, renamed `bridge_orig`. Use `tfds-directory` mode for this workflow:

1. Run `check_system.py` and ensure enough *free* space (~180 GiB for raw + processed + checkpoints).
2. Generate a download plan (does not execute the download):

   ```bash
   python3 scripts/download_bridge.py --mode tfds-directory --output-dir data/raw/bridge_v2
   ```

3. Review the plan and recommended gsutil/rclone command, then run it manually.
4. Validate the downloaded directory:

   ```bash
   python3 scripts/download_bridge.py --mode tfds-directory --output-dir data/raw/bridge_v2 --validate-only
   ```

5. Normalize records to JSONL and create deterministic splits:

   ```bash
   python3 scripts/prepare_dataset.py --input data/raw/bridge_v2/samples.jsonl --output data/processed/bridge_v2.jsonl
   python3 scripts/inspect_dataset.py --root data/processed --write-report
   ```

### Mode 2: Single archive (alternative)

When the BridgeData source URL and checksum are confirmed, download a single verified archive:

```bash
python3 scripts/download_bridge.py --mode archive --source URL --sha256 CHECKSUM --expected-gb 124
```

This tool refuses if its conservative free-space threshold is not met, resumes HTTP ranges where supported, and never re-downloads a verified archive.

Do not start `configs/bridge_ft.yaml` until a model-specific, small end-to-end smoke run has succeeded. The config is intentionally guarded with `training.allow_large_run: false`.

## Staged research workflow

The full research pipeline follows these stages. Each stage must succeed before proceeding to the next.

| Stage | Description | Where | Command |
|---|---|---|---|
| 1. Local smoke | Diagnostic pipeline validation | Local Mac | `check_system.py`, `benchmark_model.py --backend smoke`, `train.py --smoke` |
| 2. Local custom webcam | Collect and inspect local data | Local Mac | `collect_custom_data.py`, `inspect_dataset.py`, `prepare_dataset.py` |
| 3. BridgeData inspection | Validate TFDS directory, storage, inventory | Local Mac | `download_bridge.py --mode tfds-directory --validate-only` |
| 4. Remote CUDA OpenVLA | Baseline and LoRA fine-tuning on BridgeData V2 | Remote GPU | `benchmark_model.py --backend openvla`, `train.py --config configs/bridge_ft.yaml` |
| 5. Common evaluation | Identical held-out split for all models | Any | `evaluate.py --config configs/evaluation.yaml` |
| 6. Simulated deployment | Webcam demo with trained checkpoint | Local Mac | `webcam_demo.py --backend smoke` (or `--backend openvla` with local checkpoint) |

## Experiments and evaluation

All run inputs, machine information, metrics, checkpoints, and timestamps are stored under a unique experiment directory. Evaluate each of baseline / Bridge-FT / Bridge-FT+custom against an identical held-out split:

```bash
python3 scripts/evaluate.py --config configs/evaluation.yaml --predictions PATH_TO_PREDICTIONS.jsonl
```

Continuous actions use MAE/RMSE; discrete demo actions use accuracy, macro precision/recall/F1, and a confusion matrix. Report task success only when an actual evaluator supplies it.

## Webcam data and demo

With OpenCV installed:

```bash
python3 scripts/collect_custom_data.py --output data/raw/visionpilot_custom --instruction "Move toward the bottle"
python3 scripts/webcam_demo.py --instruction "Move toward the bottle" --backend smoke
```

Collector keys: `W/A/S/D` forward/left/back/right, `Q/E` look left/right, `X` stop, `Space` saves the labelled frame, `Esc` exits. The demo uses `SimulatedExecutor`; it displays predicted actions but never drives a robot.

## Why these choices

- **OpenVLA:** a credible pretrained VLA baseline; we adapt it rather than training 7B parameters from scratch.
- **BridgeData V2:** diverse real-robot demonstrations suitable for domain adaptation, subject to its license and format validation.
- **LoRA:** keeps trainable parameter count and optimizer state manageable, but it does not make a 7B model automatically fit 16 GB.
- **Apple Silicon / MLX:** useful native local tooling when a model conversion exists. This repository probes MLX but does not misrepresent an unverified OpenVLA conversion as supported.

## Limitations and next work

The full OpenVLA loader requires its official runtime and a compatible checkpoint. Full fine-tuning will almost certainly need a supported remote CUDA environment or a validated, memory-profiled MLX conversion. Add the confirmed BridgeData adapter after inspecting real samples, then implement the model-specific LoRA trainer and evaluator. No performance improvement is claimed until all three runs use a recorded common evaluation protocol.

## Citation

Please cite the official OpenVLA and BridgeData V2 papers/repositories used for the exact checkpoint and dataset release. Record their versions in each experiment manifest.
