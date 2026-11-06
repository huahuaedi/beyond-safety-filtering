import unittest

try:
    import torch
    from continuous_scene_risk_critic import ContinuousSceneRiskCritic
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class ContinuousSceneRiskCriticTest(unittest.TestCase):
    def test_shapes_and_action_gradient(self):
        critic = ContinuousSceneRiskCritic(latent_dim=16, dim=32)
        latents = torch.randn(2, 5, 4, 16)
        context = critic.scene_context(latents)
        action = torch.randn(2, 5, 24, requires_grad=True)
        risk = critic.risk(action, context)
        self.assertEqual(risk.shape, (2, 5))
        risk.mean().backward()
        self.assertGreater(float(action.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
