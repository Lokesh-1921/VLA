#!/usr/bin/env python3
"""Run the diagnostic smoke training gate; guarded against accidental large runs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from visionpilot.datasets.base import RobotSample
from visionpilot.datasets.jsonl import JsonlRobotDataset
from visionpilot.models.base import NumpySmokePolicy
from visionpilot.training.checkpointing import load_smoke_checkpoint, save_smoke_checkpoint
from visionpilot.training.trainer import SmokeTrainer
from visionpilot.utils.config import dump_config, load_config
from visionpilot.utils.device import system_report
from visionpilot.utils.logging import append_jsonl, write_json


def synthetic_samples(count: int, action_dim: int, image_size: int, seed: int) -> list[RobotSample]:
    rng = np.random.default_rng(seed)
    samples = []
    instructions = ("move left to the cup", "move right to the bottle", "move forward toward object", "stop and look left")
    for index in range(count):
        image = rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        # A deterministic target gives the diagnostic optimizer a real learnable signal.
        direction = np.array([(-1, 0), (1, 0), (0, 1), (0, 0)], dtype=np.float32)[index % 4]
        action = np.zeros(action_dim, dtype=np.float32)
        action[:2] = direction
        action[5 if action_dim > 5 else 0] = (float(image[..., 0].mean()) / 255.0 - 0.5) * 0.2
        samples.append(RobotSample(image, instructions[index % 4], action, f"synthetic-{index // 4}", index % 4, "synthetic_smoke"))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="required acknowledgement for diagnostic local training")
    parser.add_argument("--resume", help="explicit NumPy smoke checkpoint to resume")
    args = parser.parse_args()
    config = load_config(args.config)
    training, model_config, project = config["training"], config["model"], config["project"]
    is_smoke = model_config.get("backend") == "numpy_smoke" and training.get("smoke_test", False)
    if not is_smoke:
        if not training.get("allow_large_run", False):
            parser.error("Large/model-specific training is intentionally disabled. Complete and record a compatible model smoke test before setting allow_large_run: true.")
        parser.error("Only the numpy_smoke trainer is implemented in this milestone; configure the official OpenVLA LoRA backend after compatibility validation.")
    if not args.smoke:
        parser.error("Pass --smoke to acknowledge this is a diagnostic pipeline test, not OpenVLA training.")
    run_name = project.get("run_name", "smoke")
    run_id = f"{run_name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    run_dir = Path(project.get("experiment_root", "experiments")) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    samples = synthetic_samples(int(training["max_samples"]), int(model_config["action_dim"]), int(model_config["image_size"]), int(project.get("seed", 42)))
    if args.resume:
        model, resume_step = load_smoke_checkpoint(args.resume)
    else:
        model, resume_step = NumpySmokePolicy(int(model_config["action_dim"]), int(training["lora_rank"]), float(training["lora_alpha"]), int(model_config["image_size"]), int(project.get("seed", 42))), 0
    checkpoint_path = run_dir / "checkpoints" / "smoke_latest.npz"
    checkpoint_every = int(training.get("checkpoint_every_steps", 5))
    def checkpoint(step: int) -> None:
        if step % checkpoint_every == 0:
            save_smoke_checkpoint(model, checkpoint_path, step + resume_step)
    trainer = SmokeTrainer(model, float(training["learning_rate"]))
    result = trainer.train(samples, int(training["epochs"]), int(training["micro_batch_size"]), checkpoint)
    save_smoke_checkpoint(model, checkpoint_path, result.steps + resume_step)
    restored, restored_step = load_smoke_checkpoint(checkpoint_path)
    probe_delta = float(np.max(np.abs(model.predict(samples[0].image, samples[0].instruction) - restored.predict(samples[0].image, samples[0].instruction))))
    dump_config(config, run_dir / "resolved_config.yaml")
    manifest = {"run_id": run_id, "created_at": datetime.now(UTC).isoformat(), "model": model.model_id, "model_version": "diagnostic-v1", "dataset": "synthetic_smoke", "dataset_size": len(samples), "lora_rank": model.rank, "lora_alpha": model.alpha, "learning_rate": training["learning_rate"], "epochs": training["epochs"], "micro_batch_size": training["micro_batch_size"], "gradient_accumulation": training["gradient_accumulation"], "optimizer": "manual_sgd", "precision": "float32", "hardware": system_report(_bootstrap.ROOT), "checkpoint": str(checkpoint_path), "resume_step": resume_step}
    metrics = {"training_loss": result.final_loss, "training_steps": result.steps, "checkpoint_restore_step": restored_step, "checkpoint_prediction_max_abs_delta": probe_delta, "smoke_passed": bool(probe_delta == 0.0 and result.steps > 0)}
    write_json(run_dir / "manifest.json", manifest); write_json(run_dir / "metrics.json", metrics)
    for step, loss in enumerate(result.losses, 1):
        append_jsonl(run_dir / "metrics.jsonl", {"step": step + resume_step, "training_loss": loss})
    print(f"SMOKE PASSED: {metrics['smoke_passed']} | final loss={result.final_loss:.6f} | checkpoint={checkpoint_path}")
    return 0 if metrics["smoke_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
