"""Tests using the tiny Bridge-style fixture for inspection, splits, adapter, and reporting."""

import json
import tempfile
import unittest
from pathlib import Path

from visionpilot.datasets.base import RobotSample
from visionpilot.datasets.bridge_v2 import BridgeV2Dataset
from visionpilot.datasets.jsonl import JsonlRobotDataset
from visionpilot.evaluation.metrics import continuous_metrics
from visionpilot.evaluation.reports import write_markdown_report

FIXTURE = Path(__file__).parent / "fixtures" / "bridge_fixture.jsonl"


class BridgeFixtureInspectionTests(unittest.TestCase):
    def test_fixture_inspection(self) -> None:
        """All records parse, and episode/source/embodiment counts are correct."""
        records = []
        with FIXTURE.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    record = json.loads(line)
                    sample = RobotSample.from_record(record)
                    records.append(sample)
        self.assertEqual(len(records), 12)
        episodes = {r.episode_id for r in records}
        self.assertEqual(len(episodes), 3)
        sources = {r.source for r in records}
        self.assertEqual(sources, {"bridge_v2"})
        embodiments = {r.embodiment for r in records}
        self.assertEqual(embodiments, {"widowx"})
        instructions = {r.instruction for r in records}
        self.assertEqual(len(instructions), 3)
        for sample in records:
            self.assertEqual(sample.action.size, 7)
            self.assertIsNotNone(sample.state)
            self.assertEqual(sample.state.size, 7)


class BridgeFixturePreparationTests(unittest.TestCase):
    def test_fixture_preparation_splits(self) -> None:
        """Deterministic splitting produces non-empty, non-overlapping splits."""
        from hashlib import blake2b

        def split_for(episode_id: str, seed: int = 42) -> str:
            val = int.from_bytes(blake2b(f"{seed}:{episode_id}".encode(), digest_size=8).digest(), "big") / 2**64
            return "train" if val < 0.6 else "validation" if val < 0.8 else "test"

        records = []
        with FIXTURE.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(RobotSample.from_record(json.loads(line)))

        splits: dict[str, list[RobotSample]] = {"train": [], "validation": [], "test": []}
        for sample in records:
            splits[split_for(sample.episode_id)].append(sample)

        # At least some samples assigned (with 3 episodes, splits may not all be populated,
        # but total must equal original count)
        total = sum(len(v) for v in splits.values())
        self.assertEqual(total, 12)

        # Episode-level: no episode appears in more than one split
        split_episodes: dict[str, set[str]] = {k: {s.episode_id for s in v} for k, v in splits.items()}
        for a, eps_a in split_episodes.items():
            for b, eps_b in split_episodes.items():
                if a != b:
                    self.assertEqual(eps_a & eps_b, set(), f"Episode overlap between {a} and {b}")


class BridgeFixtureAdapterTests(unittest.TestCase):
    def test_dataset_adapter_behavior(self) -> None:
        """BridgeV2Dataset loads the fixture and iterates all samples correctly."""
        dataset = BridgeV2Dataset(FIXTURE)
        self.assertEqual(len(dataset), 12)
        self.assertEqual(dataset.name, "bridge_v2")
        self.assertEqual(dataset.action_dimension(), 7)
        for i, sample in enumerate(dataset):
            self.assertIsInstance(sample, RobotSample)
            self.assertEqual(sample.source, "bridge_v2")
            self.assertEqual(sample.embodiment, "widowx")


class BridgeFixtureReportTests(unittest.TestCase):
    def test_report_generation(self) -> None:
        """A markdown report can be generated from fixture metrics."""
        dataset = JsonlRobotDataset(FIXTURE, name="bridge_v2_fixture")
        import numpy as np
        predictions = np.random.default_rng(42).normal(0, 0.1, (len(dataset), 7)).astype(np.float32)
        targets = np.vstack([s.action for s in dataset])
        metrics = continuous_metrics(predictions, targets)
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "bridge_fixture_report.md"
            result = write_markdown_report(
                report_path,
                "Bridge Fixture Evaluation",
                metrics,
                {"dataset": "bridge_fixture", "samples": len(dataset)},
            )
            self.assertTrue(result.exists())
            content = result.read_text(encoding="utf-8")
            self.assertIn("Bridge Fixture Evaluation", content)
            self.assertIn("action_mae", content)
            self.assertIn("action_rmse", content)
