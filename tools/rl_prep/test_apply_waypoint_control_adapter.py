import unittest

import numpy as np

try:
    import torch  # noqa: F401
    from apply_waypoint_control_adapter import infer_initial_state
    from differentiable_vehicle_model import VehicleModelConfig
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class ApplyWaypointControlAdapterTest(unittest.TestCase):
    def test_initial_state_proxy_for_straight_constant_speed(self):
        relative = np.zeros((1, 8, 3), dtype=np.float32)
        relative[0, :, 0] = 2.0
        initial = infer_initial_state(relative, VehicleModelConfig())
        self.assertEqual(initial.shape, (1, 5))
        self.assertAlmostEqual(float(initial[0, 3]), 4.0, places=5)
        self.assertAlmostEqual(float(initial[0, 4]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
