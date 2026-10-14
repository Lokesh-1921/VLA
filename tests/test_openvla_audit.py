"""Audit tests for the OpenVLA adapter: preflight, safety guards, and API contract."""

import unittest

from visionpilot.models.openvla_adapter import OpenVLAAdapter


class OpenVLAPreflightTests(unittest.TestCase):
    def test_preflight_report_without_deps(self) -> None:
        """Preflight correctly identifies missing dependencies when torch is absent."""
        report = OpenVLAAdapter.preflight_report(
            model_id="openvla/openvla-7b",
            device="cpu",
            memory_bytes=64 * 1024**3,
        )
        required_keys = {
            "model_id", "dependencies", "device", "device_supported",
            "memory_bytes", "memory_sufficient_inference", "memory_sufficient_training",
            "checkpoint_found", "verdict", "issues",
        }
        self.assertTrue(required_keys.issubset(report.keys()), f"Missing: {required_keys - report.keys()}")
        # If torch is not installed, verdict should reflect that
        if not report["dependencies"]["torch"]["installed"]:
            self.assertEqual(report["verdict"], "missing_deps")

    def test_preflight_report_memory_insufficient(self) -> None:
        """16 GiB is flagged as insufficient for OpenVLA inference and training."""
        report = OpenVLAAdapter.preflight_report(
            model_id="openvla/openvla-7b",
            device="mps",
            memory_bytes=16 * 1024**3,
        )
        self.assertFalse(report["memory_sufficient_inference"])
        self.assertFalse(report["memory_sufficient_training"])

    def test_availability_returns_tuple(self) -> None:
        """The existing availability() API contract is preserved."""
        result = OpenVLAAdapter.availability()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], str)

    def test_predict_without_load_raises(self) -> None:
        """Calling predict on an unloaded adapter raises RuntimeError."""
        adapter = OpenVLAAdapter()
        with self.assertRaises(RuntimeError):
            adapter.predict(None, "test instruction")
