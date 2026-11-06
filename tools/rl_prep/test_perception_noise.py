import unittest

import numpy as np

import perception_noise as P


def agents(n=4000):
    rng = np.random.default_rng(0)
    return {"pos": rng.uniform(-50, 50, (n, 2)), "vel": rng.normal(0, 3, (n, 2)), "heading": rng.uniform(-3, 3, n),
            "length": np.full(n, 4.5), "width": np.full(n, 2.0)}


class PerceptionNoiseTest(unittest.TestCase):
    def test_level_zero_is_the_identity(self):
        a = agents(10)
        self.assertIs(P.perturb_agents(a, 0, P.scene_rng("t", 0)), a)
        self.assertTrue(np.allclose(P.map_offset(0, P.scene_rng("t", 0)), 0.0))

    def test_noise_scale_miss_rate_and_size_follow_the_level(self):
        a = agents()
        for level in (1, 2, 3):
            pos_s, head_s, vel_s, size_s, miss, _ = P.LEVELS[level]
            rng = P.scene_rng("scene", level)
            keep = rng.random(len(a["pos"])) >= miss          # reproduce the draw order to align kept agents
            out = P.perturb_agents(a, level, P.scene_rng("scene", level))
            self.assertAlmostEqual(len(out["pos"]) / len(a["pos"]), 1 - miss, delta=0.03)
            err = out["pos"] - a["pos"][keep]
            self.assertAlmostEqual(float(err.std()), pos_s, delta=0.08 * pos_s + 0.02)
            self.assertAlmostEqual(float((out["heading"] - a["heading"][keep]).std()), head_s, delta=0.1 * head_s + 0.01)
            self.assertAlmostEqual(float((out["width"] / 2.0).std()), size_s, delta=0.1 * size_s + 0.01)

    def test_deterministic_per_scene_and_different_across_scenes(self):
        a = agents(50)
        x = P.perturb_agents(a, 2, P.scene_rng("a", 2))["pos"]
        y = P.perturb_agents(a, 2, P.scene_rng("a", 2))["pos"]
        z = P.perturb_agents(a, 2, P.scene_rng("b", 2))["pos"]
        self.assertTrue(np.array_equal(x, y))
        self.assertFalse(x.shape == z.shape and np.allclose(x, z))

    def test_everything_missed_returns_none(self):
        P.LEVELS[9] = (0, 0, 0, 0, 1.0, 0)
        try:
            self.assertIsNone(P.perturb_agents(agents(5), 9, P.scene_rng("t", 9)))
        finally:
            del P.LEVELS[9]

    def test_shifted_predicate_moves_the_boundary(self):
        inside = lambda pts: np.asarray(pts)[..., 1] < 3.0          # drivable below y = 3
        shifted = P.shift_predicate(inside, [0.0, 1.0])               # the map appears 1 m lower
        pts = np.array([[0.0, 2.5], [0.0, 1.5]])
        self.assertEqual(inside(pts).tolist(), [True, True])
        self.assertEqual(shifted(pts).tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
