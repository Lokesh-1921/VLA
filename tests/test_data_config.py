import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from visionpilot.datasets.base import RobotSample
from visionpilot.datasets.jsonl import JsonlRobotDataset
from visionpilot.datasets.manager import DatasetManager, StoragePlan
from visionpilot.utils.config import load_config


class DataConfigTests(unittest.TestCase):
    def test_robot_sample_round_trip(self) -> None:
        original = RobotSample(None, "pick the cup", [1, 2, 3], "episode-1", 2, "test", metadata={"camera": "front"})
        self.assertEqual(RobotSample.from_record(original.to_record()).to_record(), original.to_record())

    def test_jsonl_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            path.write_text(json.dumps(RobotSample(None, "stop", [0], "e", 0, "test").to_record()) + "\n", encoding="utf-8")
            dataset = JsonlRobotDataset(path)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.action_dimension(), 1)

    def test_extending_configuration(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "mac_mlx.yaml")
        self.assertEqual(config["model"]["backend"], "numpy_smoke")
        self.assertEqual(config["training"]["micro_batch_size"], 1)
        self.assertEqual(config["project"]["seed"], 42)

    def test_safe_archive_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as payload:
                payload.writestr("nested/record.txt", "sample")
            destination = root / "out"
            extracted = DatasetManager(root).extract(archive, destination)
            self.assertEqual(len(extracted), 1)
            self.assertEqual((destination / "nested" / "record.txt").read_text(), "sample")
            self.assertGreater(StoragePlan(1024).required_free_bytes, 1024)
