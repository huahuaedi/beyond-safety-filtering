import unittest

try:
    import torch
    from continuous_scene_outcome_critic import ContinuousSceneOutcomeCritic
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class ContinuousSceneOutcomeCriticTest(unittest.TestCase):
    def test_shapes_ranges_and_action_gradient(self):
        model = ContinuousSceneOutcomeCritic(latent_dim=16, dim=32)
        context = model.scene_context(torch.randn(2, 5, 4, 16))
        action = torch.randn(2, 5, 24, requires_grad=True)
        output = model.outcomes(action, context)
        self.assertEqual(output["risk"].shape, (2, 5, 4))
        self.assertEqual(output["progress"].shape, (2, 5))
        self.assertTrue(torch.all((output["risk"] >= 0) & (output["risk"] <= 1)))
        self.assertTrue(torch.all((output["progress"] >= 0) & (output["progress"] <= 1)))
        (output["risk"].mean() + output["progress"].mean()).backward()
        self.assertGreater(float(action.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
