import unittest

from visionpilot.utils.probes import probe_mlx, probe_openvla_readiness, probe_pytorch_mps


class ProbePyTorchTests(unittest.TestCase):
    def test_pytorch_probe_structure(self) -> None:
        """Return dict has required keys and correct types regardless of install state."""
        result = probe_pytorch_mps()
        required_keys = {"installed", "version", "mps_built", "mps_available", "cuda_available", "openvla_compatible", "detail"}
        self.assertTrue(required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}")
        self.assertIsInstance(result["installed"], bool)
        self.assertIsInstance(result["mps_built"], bool)
        self.assertIsInstance(result["mps_available"], bool)
        self.assertIsInstance(result["openvla_compatible"], bool)
        self.assertIsInstance(result["detail"], str)


class ProbeMLXTests(unittest.TestCase):
    def test_mlx_probe_never_claims_openvla(self) -> None:
        """MLX probe must always report openvla_compatible=False."""
        result = probe_mlx()
        self.assertFalse(result["openvla_compatible"])
        required_keys = {"installed", "version", "openvla_compatible", "detail"}
        self.assertTrue(required_keys.issubset(result.keys()))


class ProbeOpenVLAReadinessTests(unittest.TestCase):
    def test_openvla_readiness_report_structure(self) -> None:
        """Report contains all required fields."""
        result = probe_openvla_readiness()
        required = {
            "model_id", "dependencies", "device", "device_supported",
            "memory_bytes", "memory_sufficient_inference", "memory_sufficient_training",
            "checkpoint_found", "checkpoint_detail", "verdict", "issues",
        }
        self.assertTrue(required.issubset(result.keys()), f"Missing: {required - result.keys()}")
        self.assertIsInstance(result["dependencies"], dict)
        self.assertIsInstance(result["issues"], list)
        self.assertIn(result["verdict"], ("ready", "missing_deps", "missing_checkpoint", "insufficient_memory", "unsupported_device", "not_ready"))

    def test_openvla_readiness_memory_check(self) -> None:
        """16 GiB should be flagged as insufficient for OpenVLA."""
        mem_16g = 16 * 1024**3
        result = probe_openvla_readiness(memory_bytes=mem_16g)
        self.assertFalse(result["memory_sufficient_inference"])
        self.assertFalse(result["memory_sufficient_training"])
