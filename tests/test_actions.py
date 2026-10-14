import unittest

import numpy as np

from visionpilot.actions.adapter import ContinuousToDiscreteAdapter
from visionpilot.actions.executor import SimulatedExecutor
from visionpilot.actions.schema import ContinuousAction, DiscreteAction


class ActionTests(unittest.TestCase):
    def test_adapter_preserves_demo_mapping(self) -> None:
        adapter = ContinuousToDiscreteAdapter()
        self.assertEqual(adapter.adapt(ContinuousAction(np.array([-0.8, 0.1]))), DiscreteAction.MOVE_LEFT)
        self.assertEqual(adapter.adapt(ContinuousAction(np.array([0.0, 0.6]))), DiscreteAction.MOVE_FORWARD)
        self.assertEqual(adapter.adapt(ContinuousAction(np.zeros(7))), DiscreteAction.STOP)

    def test_simulated_executor_rate_and_stop(self) -> None:
        executor = SimulatedExecutor(min_confidence=0.5, max_actions_per_second=1000)
        self.assertTrue(executor.execute(DiscreteAction.MOVE_LEFT, 0.9))
        self.assertFalse(executor.execute(DiscreteAction.MOVE_LEFT, 0.1))
        executor.emergency_stop()
        self.assertFalse(executor.execute(DiscreteAction.MOVE_LEFT, 0.9))
