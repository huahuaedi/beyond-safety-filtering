import unittest

try:
    import numpy as np
    import torch
    from build_pseudo_control_sidecars import estimate_pseudo_controls, reconstruct_and_refine
    from differentiable_vehicle_model import DifferentiableVehicleModel, VehicleModelConfig
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class PseudoControlTest(unittest.TestCase):
    def test_constant_speed_straight_trajectory_is_reconstructed(self):
        config = VehicleModelConfig()
        initial = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]])
        controls = torch.zeros(1, 8, 2)
        poses = DifferentiableVehicleModel(config)(initial, controls)[..., :3].numpy()
        estimated_initial, estimated_controls, clipped = estimate_pseudo_controls(
            poses, config.action_dt, config
        )
        _, reconstructed, position_rmse, heading_rmse = reconstruct_and_refine(
            poses, estimated_initial, estimated_controls, config,
            steps=0, learning_rate=0.05, regularization=0.01,
        )
        self.assertFalse(clipped.any())
        np.testing.assert_allclose(reconstructed, poses, atol=2e-3)
        self.assertLess(float(position_rmse[0]), 2e-3)
        self.assertLess(float(heading_rmse[0]), 1e-5)


if __name__ == "__main__":
    unittest.main()

