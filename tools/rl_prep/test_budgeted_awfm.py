import unittest
import numpy as np
import budgeted_awfm as B


def scores(passes, progress=None):
    n = len(passes)
    p = np.asarray(passes, dtype=np.float32)
    return {"no_at_fault_collisions": p, "drivable_area_compliance": np.ones(n, np.float32),
            "time_to_collision_within_bound": np.ones(n, np.float32),
            "ego_progress": np.linspace(0, 1, n, dtype=np.float32) if progress is None else np.asarray(progress, np.float32)}


class TiltTest(unittest.TestCase):
    def test_kl_hits_budget_and_is_monotone(self):
        r = B.scene_reward(scores([1] * 4 + [0] * 60))
        prev = 0.0
        for eps in (0.05, 0.2, 0.5, 1.0):
            w, tau, kl = B.tilt_weights(r, eps)
            self.assertAlmostEqual(kl, eps, places=5)
            self.assertAlmostEqual(w.sum(), 1.0, places=9)
            q = float(w[:4].sum())
            self.assertGreater(q, prev)
            prev = q

    def test_zero_budget_and_constant_reward_are_uniform(self):
        for r, eps in ((np.array([1., 0, 0]), 0.0), (np.ones(5), 0.7)):
            w, tau, kl = B.tilt_weights(r, eps)
            np.testing.assert_allclose(w, np.full(len(r), 1 / len(r)))
            self.assertEqual(kl, 0.0)

    def test_budget_above_reachable_concentrates_on_argmax_set(self):
        r = np.array([1., 1., 0., 0., 0., 0., 0., 0.])
        w, tau, kl = B.tilt_weights(r, 5.0)
        np.testing.assert_allclose(w[:2], 0.5)
        self.assertAlmostEqual(kl, np.log(8 / 2), places=6)

    def test_promised_pass_matches_closed_form_for_binary_gate(self):
        s = scores([1] * 8 + [0] * 56)
        p, q, kl, tau = B.promised_pass(s, 0.3)
        self.assertAlmostEqual(p, 8 / 64)
        e = 1 / tau
        self.assertAlmostEqual(q, 8 * np.exp(e) / (8 * np.exp(e) + 56), places=6)
        self.assertGreater(q, p)

    def test_no_passer_scene_has_no_gain(self):
        p, q, kl, tau = B.promised_pass(scores([0] * 10), 0.5)
        self.assertEqual((p, q), (0.0, 0.0))

    def test_utility_term_only_reorders_passers(self):
        s = scores([1, 1, 1, 0], progress=[0.1, 0.5, 0.9, 1.0])
        r = B.scene_reward(s, utility_weight=1.0)
        self.assertLess(r[0], r[1]); self.assertLess(r[1], r[2])
        self.assertGreater(r[:3].min(), r[3] + 0.3)


class PreserveTest(unittest.TestCase):
    def test_passers_only_vs_all_and_determinism(self):
        s = scores([0, 0, 1, 0, 1, 0])
        picks = {B.preserve_candidate(s, "passers", k) for k in range(40)}
        self.assertEqual(picks, {2, 4})
        self.assertTrue({B.preserve_candidate(s, "all", k) for k in range(80)} - {2, 4})
        self.assertEqual(B.preserve_candidate(s, "all", 3), B.preserve_candidate(s, "all", 3))

    def test_no_passer_and_unknown_kind(self):
        self.assertIsNone(B.preserve_candidate(scores([0, 0, 0]), "passers", 0))
        with self.assertRaises(ValueError):
            B.preserve_candidate(scores([1]), "bogus", 0)


class ScheduleTest(unittest.TestCase):
    def test_deterministic_skips_and_follows_weights(self):
        recs = [{"token": "a"}, {"token": "b"}, {"token": "skip"}]
        w = {"a": np.array([0.0, 1.0, 0.0]), "b": np.array([0.5, 0.5, 0.0]), "skip": None}
        s1 = B.build_tilted_schedule(recs, lambda r: w[r["token"]], candidates_per_scene=6, epochs=2, seed=1)
        s2 = B.build_tilted_schedule(recs, lambda r: w[r["token"]], candidates_per_scene=6, epochs=2, seed=1)
        self.assertEqual([(r["token"], c, e) for r, c, e in s1], [(r["token"], c, e) for r, c, e in s2])
        self.assertEqual(len(s1), 2 * 2 * 6)
        self.assertTrue(all(c == 1 for r, c, e in s1 if r["token"] == "a"))
        self.assertTrue(all(c in (0, 1) for r, c, e in s1 if r["token"] == "b"))
        self.assertFalse(any(r["token"] == "skip" for r, c, e in s1))

    def test_rejects_non_probability(self):
        with self.assertRaises(ValueError):
            B.build_tilted_schedule([{"token": "a"}], lambda r: np.array([0.4, 0.4]), candidates_per_scene=1, epochs=1, seed=0)


class TransferTest(unittest.TestCase):
    def test_bins_without_unmatched_efficiency(self):
        p = np.array([0.0, 0.01, 0.5, 0.5, 1.0])
        promised = np.array([0.0, 0.1, 0.2, 0.2, 0.0])
        fix = np.array([0, 1, 1, 0, 0])
        new = np.array([0, 0, 0, 1, 0])
        rows = {(r["p_lo"], r["p_hi"]): r for r in B.transfer_table(p, promised, fix, new)}
        mid = rows[(0.3, 0.7)]
        self.assertEqual(mid["n"], 2)
        self.assertAlmostEqual(mid["target_pool_pass_gain"], 0.2)
        self.assertAlmostEqual(mid["realised"], 0.0)
        self.assertNotIn("efficiency", mid)
        self.assertAlmostEqual(rows[(0.0, 0.02)]["realised"], 0.5)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            B.transfer_table([0.1], [0.1, 0.2], [0], [0])


if __name__ == "__main__":
    unittest.main()
