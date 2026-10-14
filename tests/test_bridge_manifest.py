"""Tests for the BridgeData V2 TFDS manifest, inventory, and download plan tools."""

import tempfile
import unittest
from pathlib import Path

from visionpilot.datasets.bridge_manifest import (
    BridgeManifest,
    estimate_storage_requirements,
    generate_download_plan,
    scan_bridge_directory,
    validate_bridge_inventory,
)


class BridgeManifestScanTests(unittest.TestCase):
    def test_scan_empty_directory(self) -> None:
        """Scanning an empty directory returns an empty inventory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inventory = scan_bridge_directory(tmpdir)
            self.assertEqual(inventory.file_count, 0)
            self.assertEqual(inventory.total_bytes, 0)
            self.assertEqual(len(inventory.files), 0)

    def test_scan_populated_fixture(self) -> None:
        """Scanning a directory with dummy shard files catalogues them correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create a few dummy files matching expected patterns
            (root / "dataset_info.json").write_text("{}", encoding="utf-8")
            (root / "features.json").write_text("{}", encoding="utf-8")
            shard = root / "bridge_orig-train.tfrecord-00000-of-00256"
            shard.write_bytes(b"x" * 100)

            inventory = scan_bridge_directory(root)
            self.assertEqual(inventory.file_count, 3)
            self.assertGreater(inventory.total_bytes, 0)
            paths = inventory.relative_paths()
            self.assertIn("dataset_info.json", paths)
            self.assertIn("features.json", paths)


class BridgeManifestValidationTests(unittest.TestCase):
    def test_validate_complete_inventory(self) -> None:
        """A directory with all expected files reports complete."""
        manifest = BridgeManifest.from_defaults()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for filename in manifest.all_expected_files():
                filepath = root / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_bytes(b"x")

            inventory = scan_bridge_directory(root)
            report = validate_bridge_inventory(inventory, manifest)
            self.assertTrue(report.complete)
            self.assertEqual(len(report.missing_files), 0)

    def test_validate_incomplete_inventory(self) -> None:
        """Missing files are flagged in the validation report."""
        manifest = BridgeManifest.from_defaults()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Only create top-level metadata, not shards
            (root / "dataset_info.json").write_text("{}", encoding="utf-8")

            inventory = scan_bridge_directory(root)
            report = validate_bridge_inventory(inventory, manifest)
            self.assertFalse(report.complete)
            self.assertGreater(len(report.missing_files), 0)
            self.assertIn("features.json", report.missing_files)


class BridgeStorageEstimateTests(unittest.TestCase):
    def test_storage_estimate_positive(self) -> None:
        """Storage estimates are all positive."""
        estimates = estimate_storage_requirements()
        self.assertGreater(estimates["raw_dataset_bytes"], 0)
        self.assertGreater(estimates["processed_cache_bytes"], 0)
        self.assertGreater(estimates["checkpoint_bytes"], 0)
        self.assertGreater(estimates["total_estimated_bytes"], 0)
        self.assertGreater(estimates["total_estimated_gib"], 0)


class BridgeDownloadPlanTests(unittest.TestCase):
    def test_download_plan_empty_directory(self) -> None:
        """Download plan for empty directory lists all files as needed."""
        manifest = BridgeManifest.from_defaults()
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = generate_download_plan(tmpdir, manifest)
            self.assertGreater(len(plan.files_needed), 0)
            self.assertEqual(plan.files_already_present, 0)
            self.assertIn("gsutil", plan.recommended_command)
