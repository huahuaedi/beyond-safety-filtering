import tempfile
import unittest
from pathlib import Path

import numpy as np

import probe_policy_context as P


class ProbeTest(unittest.TestCase):
    def test_ridge_recovers_linear_signal_and_r2(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 8)); y = x[:, 0] * 2 + 0.1 * rng.normal(size=300)
        f = P.ridge_fit(x, y, 1.0)
        self.assertGreater(P.r2(y, f(x)), 0.95)
        noise = rng.normal(size=300)
        g = P.ridge_fit(x[:200], noise[:200], 1000.0)
        self.assertLess(P.r2(noise[200:], g(x[200:])), 0.1)

    def test_targets_from_candidate_file(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "s.npz"
            np.savez(p, score__drivable_area_compliance=np.array([1, 0, 0, 1.]), score__ego_progress=np.array([.2, .4, .6, .8]),
                     reference_probe__drivable_area_compliance=np.array([1., 0.]))
            r = P.targets(p)
            self.assertAlmostEqual(r["offroad_share"], 0.5)
            self.assertEqual(r["ref_dac_fail"], 1.0)
            self.assertAlmostEqual(r["mean_progress"], 0.5)

    def test_log_bootstrap_interval_brackets_estimate(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=400); p = y + 0.5 * rng.normal(size=400); logs = [i % 20 for i in range(400)]
        lo, hi = P.boot_by_log(P.r2, y, p, logs, rng, n=200)
        self.assertLess(lo, P.r2(y, p) + 1e-9); self.assertGreater(hi, P.r2(y, p) - 1e-9)


if __name__ == "__main__":
    unittest.main()
