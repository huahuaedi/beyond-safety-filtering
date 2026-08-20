import unittest

try:
    import torch
    from torch import nn
    from unified_correction_policy import UnifiedCorrectionPolicy
    from unified_correction_rl import UnifiedCorrectionRL, lower_tail_cvar
except ImportError:
    torch = None


class DummyOutcomeCritic(nn.Module if torch is not None else object):
    def outcomes(self, action, context):
        distance = action[..., 0::3].sum(-1)
        curvature = action[..., 2::3].abs().mean(-1)
        risk = torch.stack([curvature, curvature, curvature, curvature], -1).sigmoid()
        return {"risk": risk, "progress": distance.sigmoid()}


@unittest.skipIf(torch is None, "PyTorch unavailable")
class UnifiedCorrectionRLTest(unittest.TestCase):
    def test_cvar(self):
        value = lower_tail_cvar(torch.tensor([-2.0, -1.0, 1.0, 3.0]), 0.5)
        self.assertAlmostEqual(float(value), -1.5)

    def test_end_to_end_gradient(self):
        policy = UnifiedCorrectionPolicy(dim=24, layers=1, heads=4, wm_context_dim=16)
        model = UnifiedCorrectionRL(policy, DummyOutcomeCritic())
        proposal = torch.zeros(4, 8, 3)
        proposal[..., 0] = 1.0
        output = model(
            proposal, torch.randn(4, 16), sample_correction=True
        )
        self.assertEqual(output["reward"].shape, (4,))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in policy.parameters() if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)

    def test_precomputed_projection_matches_internal_projection(self):
        # A caller (e.g. train_policy()'s RL loop) that already ran
        # policy.project() on this exact proposal/initial_state must get the
        # bit-identical result passing it in via `projected=` as it would
        # letting forward() recompute it -- otherwise sharing the projection
        # to avoid a second LBFGS solve would silently change behavior
        # instead of just saving time.
        policy = UnifiedCorrectionPolicy(dim=24, layers=1, heads=4, wm_context_dim=16)
        model = UnifiedCorrectionRL(policy, DummyOutcomeCritic()).eval()
        proposal = torch.zeros(4, 8, 3)
        proposal[..., 0] = 1.0
        context = torch.randn(4, 16)
        with torch.no_grad():
            recomputed = model(proposal, context, sample_correction=False)
            projected = policy.project(proposal)
            shared = model(
                proposal, context, projected=projected, sample_correction=False,
            )
        self.assertTrue(
            torch.allclose(shared["corrected_relative"], recomputed["corrected_relative"])
        )
        self.assertTrue(
            torch.allclose(shared["reference_relative"], recomputed["reference_relative"])
        )


if __name__ == "__main__":
    unittest.main()
