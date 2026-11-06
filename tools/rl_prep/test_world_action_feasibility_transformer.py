import unittest

try:
    import torch
    from train_world_action_feasibility_transformer import FeasibilityTransformer, MODES
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class FeasibilityTransformerTest(unittest.TestCase):
    def test_all_ablation_modes_preserve_scene_and_candidate_axes(self):
        action = torch.randn(2, 64, 24)
        latent = torch.randn(2, 64, 4, 4096)
        condition = torch.randn(2, 64, 11)
        for mode in MODES:
            model = FeasibilityTransformer(mode, dim=64).eval()
            with torch.inference_mode():
                classification, regression = model(action, latent, condition)
            self.assertEqual(classification.shape, (2, 64, 8))
            self.assertEqual(regression.shape, (2, 64, 8))

    def test_full_model_is_candidate_permutation_equivariant(self):
        torch.manual_seed(1)
        model = FeasibilityTransformer("full", dim=64).eval()
        action = torch.randn(1, 64, 24)
        latent = torch.randn(1, 64, 4, 4096)
        condition = torch.randn(1, 64, 11)
        permutation = torch.randperm(64)
        with torch.inference_mode():
            original = model(action, latent, condition)[0]
            permuted = model(
                action[:, permutation], latent[:, permutation], condition[:, permutation]
            )[0]
        torch.testing.assert_close(permuted, original[:, permutation], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
