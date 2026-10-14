import tempfile
import unittest
from pathlib import Path

import numpy as np

from visionpilot.datasets.base import RobotSample
from visionpilot.inference.pipeline import InferencePipeline
from visionpilot.models.base import NumpySmokePolicy
from visionpilot.training.checkpointing import load_smoke_checkpoint, save_smoke_checkpoint
from visionpilot.training.trainer import SmokeTrainer


class ModelTrainingTests(unittest.TestCase):
    def samples(self) -> list[RobotSample]:
        return [
            RobotSample(np.full((8, 8, 3), fill_value=i * 30, dtype=np.uint8), "move left" if i % 2 else "move right", [(-1) ** i, 0, 0, 0, 0, 0, 0], f"e-{i}", 0, "test")
            for i in range(8)
        ]

    def test_training_changes_lora_and_checkpoint_roundtrips(self) -> None:
        model = NumpySmokePolicy(image_size=8, rank=2)
        result = SmokeTrainer(model).train(self.samples(), epochs=2)
        self.assertGreater(result.steps, 0)
        self.assertGreater(float(np.abs(model.lora_a).sum()), 0.0)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "smoke.npz"
            save_smoke_checkpoint(model, checkpoint, result.steps)
            restored, step = load_smoke_checkpoint(checkpoint)
            self.assertEqual(step, result.steps)
            original = model.predict(self.samples()[0].image, "move left")
            self.assertTrue(np.allclose(original, restored.predict(self.samples()[0].image, "move left")))

    def test_inference_pipeline(self) -> None:
        result = InferencePipeline(NumpySmokePolicy(image_size=8)).infer(np.zeros((8, 8, 3), dtype=np.uint8), "stop")
        self.assertEqual(result.action.values.size, 7)
        self.assertGreaterEqual(result.confidence, 0.0)
