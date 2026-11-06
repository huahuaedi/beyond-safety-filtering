import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import wm_action_sensitivity as sens

S, C, D = 36, 40, 48


def synth(action_gain, noise=0.05, hidden_gain=0.0, seed=0):
    """Latents = scene identity + action_gain * (A @ W) + hidden consequence + noise."""
    rng = np.random.default_rng(seed)
    actions = rng.normal(size=(S, C, 24)).astype(np.float32)
    scene = rng.normal(size=(S, 1, D)) * 3.0
    projection = rng.normal(size=(24, D)) / np.sqrt(24)
    hidden = rng.normal(size=(S, C))
    direction = rng.normal(size=(D,))
    z = scene + action_gain * (actions @ projection) + hidden_gain * hidden[..., None] * direction
    z = z + noise * rng.normal(size=(S, C, D))
    return actions, z.astype(np.float32), hidden


class HelperTest(unittest.TestCase):
    def test_rankdata_ties_and_spearman(self):
        np.testing.assert_allclose(sens.rankdata([10, 20, 20, 5]), [2.0, 3.5, 3.5, 1.0])
        x = np.arange(20.0)
        self.assertAlmostEqual(sens.spearman(x, x ** 3), 1.0)
        self.assertAlmostEqual(sens.spearman(x, -x), -1.0)

    def test_pairwise_l2_matches_bruteforce(self):
        x = np.random.default_rng(1).normal(size=(7, 5))
        brute = [np.linalg.norm(x[i] - x[j]) for i in range(7) for j in range(i + 1, 7)]
        np.testing.assert_allclose(sens.pairwise_l2(x), brute, atol=1e-9)

    def test_partial_correlation_removes_shared_component(self):
        self.assertAlmostEqual(sens.partial_from_rhos(0.6, 0.8, 0.75), 0.0, places=6)
        self.assertGreater(sens.partial_from_rhos(0.6, 0.1, 0.1), 0.5)

    def test_variance_fraction_extremes(self):
        rng = np.random.default_rng(2)
        scene_only = np.repeat(rng.normal(size=(S, 1, D)), C, axis=1)
        self.assertAlmostEqual(sens.within_scene_variance_fraction(scene_only), 0.0, places=9)
        no_scene = rng.normal(size=(S, C, D))
        self.assertGreater(sens.within_scene_variance_fraction(no_scene), 0.9)

    def test_effective_rank(self):
        rng = np.random.default_rng(3)
        rank2 = rng.normal(size=(500, 2)) @ rng.normal(size=(2, 30))
        self.assertLess(sens.effective_rank(rank2), 2.5)
        self.assertGreater(sens.effective_rank(rng.normal(size=(500, 30))), 20.0)


class AnalysisTest(unittest.TestCase):
    def test_action_driven_latent_is_detected(self):
        actions, z, hidden = synth(action_gain=1.0)
        result = sens.analyse_layer(actions, z, hidden, seed=0)
        self.assertGreater(result["action_latent_distance_spearman"]["mean"], 0.6)
        self.assertLess(abs(result["action_latent_distance_spearman_shuffled_null"]["mean"]), 0.1)
        self.assertGreater(result["action_decodability_r2"]["r2"], 0.8)
        self.assertGreater(result["latent_from_action_r2"]["linear"]["r2"], 0.8)
        self.assertLess(result["near_pair_latent_ratio"]["median"], 0.9)

    def test_action_blind_latent_is_detected(self):
        actions, z, hidden = synth(action_gain=0.0, noise=1.0)
        result = sens.analyse_layer(actions, z, hidden, seed=0)
        self.assertLess(abs(result["action_latent_distance_spearman"]["mean"]), 0.15)
        self.assertLess(result["action_decodability_r2"]["r2"], 0.15)
        self.assertLess(result["latent_from_action_r2"]["linear"]["r2"], 0.15)
        self.assertGreater(result["near_pair_latent_ratio"]["mean"], 0.9)

    def test_partial_outcome_signal_needs_more_than_action(self):
        # The outcome depends on a hidden consequence that only the latent encodes.
        actions, z, hidden = synth(action_gain=1.0, hidden_gain=2.0, noise=0.05)
        with_signal = sens.analyse_layer(actions, z, hidden, seed=0)["outcome_gap"]
        self.assertGreater(with_signal["latent_partial_given_action"]["mean"], 0.2)
        # Same latent structure but the outcome is unrelated to the latent hidden part.
        rng = np.random.default_rng(9)
        unrelated = sens.analyse_layer(actions, z, rng.normal(size=hidden.shape), seed=0)["outcome_gap"]
        self.assertLess(abs(unrelated["latent_partial_given_action"]["mean"]), 0.1)

    def test_quadratic_basis_captures_nonlinear_action_dependence(self):
        rng = np.random.default_rng(4)
        actions = rng.normal(size=(S, C, 6)).astype(np.float32)
        z = np.concatenate([actions[..., :3] ** 2, actions[..., 3:5] * actions[..., 4:6]], axis=-1)
        z = z - z.mean(axis=1, keepdims=True)
        a = actions - actions.mean(axis=1, keepdims=True)
        linear = sens.held_out_ridge_r2(a, z, seed=0)["r2"]
        quadratic = sens.held_out_ridge_r2(sens.quadratic_features(a), z, seed=0)["r2"]
        self.assertGreater(quadratic, 0.8)
        self.assertGreater(quadratic, linear + 0.4)


class CliTest(unittest.TestCase):
    def test_end_to_end_and_no_overwrite(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(12):
                split = "train" if i < 8 else "val"
                (root / "data" / split).mkdir(parents=True, exist_ok=True)
                (root / "lat" / split).mkdir(parents=True, exist_ok=True)
                payload = {"relative_trajectories": rng.normal(size=(16, 8, 3)).astype("float32")}
                payload.update({k: rng.uniform(size=16) for k in sens.SCORE_KEYS})
                np.savez(root / "data" / split / f"s{i}.npz", **payload)
                np.savez(root / "lat" / split / f"s{i}.npz", eoa_latents=rng.normal(size=(16, 2, 20)).astype("float32"))
            out = root / "out.json"
            args = ["--data-root", str(root / "data"), "--latent-root", str(root / "lat"), "--output", str(out)]
            sens.main(args)
            result = json.loads(out.read_text())
            self.assertEqual(result["scenes"], 12)
            self.assertEqual(sorted(result["layers"]), ["0", "1"])
            with self.assertRaises(SystemExit):
                sens.main(args)  # never silently overwrite a result


if __name__ == "__main__":
    unittest.main()
