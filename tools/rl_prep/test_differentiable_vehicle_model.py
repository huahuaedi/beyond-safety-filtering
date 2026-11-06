import unittest

try:
    import torch
    from differentiable_vehicle_model import (
        ContinuousControlHead,
        ContinuousControlTrajectoryHead,
        DifferentiableVehicleModel,
        VehicleModelConfig,
        absolute_to_relative_se2,
        relative_to_absolute_se2,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class DifferentiableVehicleModelTest(unittest.TestCase):
    def setUp(self):
        self.config = VehicleModelConfig()

    def test_constant_speed_straight_rollout(self):
        model = DifferentiableVehicleModel(self.config)
        initial = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]])
        controls = torch.zeros(1, 8, 2)
        states = model(initial, controls)
        torch.testing.assert_close(states[0, -1, 0], torch.tensor(20.0), atol=2e-3, rtol=0)
        torch.testing.assert_close(states[0, :, 1], torch.zeros(8), atol=1e-5, rtol=0)

    def test_pose_conversion_round_trip(self):
        increments = torch.tensor(
            [[[1.0, 0.0, 0.1], [1.0, 0.1, -0.2], [0.8, -0.1, 0.05]]]
        )
        absolute = relative_to_absolute_se2(increments)
        recovered = absolute_to_relative_se2(absolute)
        torch.testing.assert_close(recovered, increments, atol=1e-5, rtol=1e-5)

    def test_controls_are_smoothly_bounded_and_zero_centered(self):
        head = ContinuousControlHead(16, self.config)
        with torch.no_grad():
            head.network[-1].weight.zero_()
            head.network[-1].bias.zero_()
        controls = head(torch.zeros(2, 8, 16))
        torch.testing.assert_close(controls, torch.zeros_like(controls), atol=1e-6, rtol=0)
        self.assertGreaterEqual(float(controls[..., 0].min().detach()), self.config.min_acceleration)
        self.assertLessEqual(float(controls[..., 0].max().detach()), self.config.max_acceleration)

    def test_risk_style_loss_reaches_action_hidden_states(self):
        torch.manual_seed(3)
        module = ContinuousControlTrajectoryHead(32, self.config)
        hidden = torch.randn(2, 8, 32, requires_grad=True)
        initial = torch.tensor([[0.0, 0.0, 0.0, 6.0, 0.0]]).expand(2, -1)
        output = module(hidden, initial)
        loss = output["absolute_trajectory"].square().mean()
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(float(hidden.grad.abs().sum()), 0.0)

    def test_invalid_discretization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            DifferentiableVehicleModel(VehicleModelConfig(integration_dt=0.3))


if __name__ == "__main__":
    unittest.main()
