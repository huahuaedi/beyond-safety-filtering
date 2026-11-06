import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from differentiable_vehicle_model import ContinuousControlTrajectoryHead
    from train_world_action_feasibility_transformer import FeasibilityTransformer
    from wm_mean_risk_critic import ENVIRONMENTAL_RISKS, WMMeanRiskCritic
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class WMMeanRiskCriticTest(unittest.TestCase):
    def test_risk_objective_backpropagates_to_action_only(self):
        critic = WMMeanRiskCritic(model_dim=32).freeze()
        action = torch.randn(2, 3, 24, requires_grad=True)
        latent = torch.randn(2, 3, 4, 4096)
        loss = critic.risk_objective(action, latent)
        loss.backward()
        self.assertIsNotNone(action.grad)
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertGreater(float(action.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))

    def test_candidate_axis_is_not_fixed_to_64(self):
        critic = WMMeanRiskCritic(model_dim=32).eval()
        output = critic(torch.randn(1, 5, 24), torch.randn(1, 5, 4, 4096))
        self.assertEqual(output.shape, (1, 5, len(ENVIRONMENTAL_RISKS)))

    def test_single_candidate_interface(self):
        critic = WMMeanRiskCritic(model_dim=32).eval()
        output = critic(torch.randn(2, 24), torch.randn(2, 4, 4096))
        self.assertEqual(output.shape, (2, len(ENVIRONMENTAL_RISKS)))

    def test_risk_gradient_crosses_control_head_and_vehicle_model(self):
        torch.manual_seed(4)
        trajectory_head = ContinuousControlTrajectoryHead(hidden_dim=32)
        critic = WMMeanRiskCritic(model_dim=32).freeze()
        hidden = torch.randn(2, 8, 32, requires_grad=True)
        initial_state = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]]).expand(2, -1)
        relative = trajectory_head(hidden, initial_state)["relative_trajectory"].flatten(-2)
        latent = torch.randn(2, 4, 4096)
        critic.risk_objective(relative, latent).backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(float(hidden.grad.abs().sum()), 0.0)

    def test_checkpoint_loader_rejects_wrong_mode(self):
        model = FeasibilityTransformer("action", dim=32)
        payload = {
            "model": model.state_dict(),
            "model_dim": 32,
            "results": {"feature_mode": "action"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "expected a wm_mean checkpoint"):
                WMMeanRiskCritic.from_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
