import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_candidate_headroom as h


def write_scene(path, nc, dac, ttc, ep, comfort, ddc, ref):
    d = {}
    for name, v in zip(h.SAFETY + h.EXTRA, (nc, dac, ttc, ep, comfort, ddc)):
        d["score__" + name] = np.asarray(v, np.float32)
        d["reference_probe__" + name] = np.asarray([ref[name]] * 2, np.float32)
    np.savez(path, **d)


class HeadroomTest(unittest.TestCase):
    def test_pdms_formula(self):
        self.assertAlmostEqual(h.pdms(1, 1, 0.5, 1, 1), (2.5 + 5 + 2) / 12)
        self.assertEqual(h.pdms(0, 1, 1, 1, 1), 0.0)

    def test_best_of_k_matches_enumeration(self):
        v = np.array([0.1, 0.9, 0.4, 0.6, 0.2])
        for k in (1, 2, 3, 5):
            brute = np.mean([max(c) for c in itertools.combinations(v, k)])
            self.assertAlmostEqual(h.expected_best_of_k(v, k), brute)

    def test_oracle_and_repairable(self):
        ones = np.ones(4)
        ref = {"no_at_fault_collisions": 1, "drivable_area_compliance": 0,
               "time_to_collision_within_bound": 1, "ego_progress": 0.5,
               "comfort": 1, "driving_direction_compliance": 1}
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "s.npz"
            write_scene(p, ones, [0, 1, 1, 0], ones, [0.9, 0.4, 0.8, 0.9], ones, ones, ref)
            cand, rf, _ = h.load_scene(p)
            r = h.scene_stats(cand, rf)
        self.assertTrue(r["ref_unsafe"])
        self.assertTrue(r["fully_safe_exists"])
        self.assertEqual(r["ref_pdms"], 0.0)  # reference fails DAC
        # DAC=0 zeroes the PDMS (multiplicative), so the 0.9-progress candidates score 0
        self.assertAlmostEqual(r["cand_max_pdms"], (5 * 0.8 + 5 + 2) / 12, places=6)
        self.assertAlmostEqual(r["oracle_strict"], (5 * 0.8 + 5 + 2) / 12, places=6)

    def test_summarize_runs(self):
        rng = np.random.default_rng(0)
        ref = {k: 1.0 for k in h.SAFETY + h.EXTRA}
        ref["ego_progress"] = 0.5
        rows = []
        with tempfile.TemporaryDirectory() as t:
            for i in range(6):
                p = Path(t) / f"{i}.npz"
                ep = rng.uniform(0.3, 1.0, 8)
                write_scene(p, np.ones(8), np.ones(8), np.ones(8), ep, np.ones(8), np.ones(8), ref)
                cand, rf, _ = h.load_scene(p)
                rows.append(h.scene_stats(cand, rf))
        s = h.summarize(rows, rng)
        self.assertEqual(s["scenes"], 6)
        gains = [s["best_of_k_gain_x100"][str(k)]["mean"] for k in h.KS]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(gains, gains[1:])))
        self.assertEqual(s["reference_unsafe_scenes"], 0)


if __name__ == "__main__":
    unittest.main()
