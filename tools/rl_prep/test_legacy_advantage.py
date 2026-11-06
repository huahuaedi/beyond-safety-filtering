import unittest
import numpy as np
from legacy_advantage import legacy_reference_anchored_advantage as legacy
from policy_optimization import compute_reference_anchored_advantage as current

KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
        "ego_progress", "driving_direction_compliance")


def group(rng, n=16, p_fail=0.2):
    s = {k: np.ones(n) for k in KEYS}
    s["ego_progress"] = rng.uniform(0.3, 1.0, n)
    for k in KEYS[:3]:
        s[k] = np.where(rng.random(n) < p_fail / 3, 0.0, 1.0)
    return s


REF = {k: 1.0 for k in KEYS}


class LegacyAdvantageTests(unittest.TestCase):
    def test_group_is_zero_mean_in_every_branch(self):
        rng = np.random.default_rng(0)
        for p in (0.0, 0.1, 0.5, 0.95, 1.0):
            for _ in range(20):
                adv, _, _ = legacy(group(rng, p_fail=p), REF)
                self.assertAlmostEqual(float(adv.mean()), 0.0, places=5)

    def test_all_feasible_group_has_unit_std(self):
        rng = np.random.default_rng(1)
        adv, feas, _ = legacy(group(rng, p_fail=0.0), REF)
        self.assertTrue(feas.all())
        self.assertAlmostEqual(float(adv.std()), 1.0, places=3)

    def test_current_version_differs_as_documented(self):
        rng = np.random.default_rng(2)
        s = group(rng, p_fail=0.5)
        a_old, f_old, _ = legacy(s, REF)
        a_new, f_new, _ = current(s, REF)
        self.assertTrue((f_old == f_new).all())              # same feasibility rule
        if (~f_new).any() and f_new.sum() >= 2:
            self.assertLess(float(a_new.mean()), -0.05)       # current: not centred
            self.assertAlmostEqual(float(a_old.mean()), 0.0, places=5)
            self.assertLessEqual(float(np.abs(a_new[f_new]).max()), 0.5 + 1e-6)

    def test_noise_floor_shrinks_signal(self):
        rng = np.random.default_rng(3)
        s = group(rng, p_fail=0.0)
        a0, _, _ = legacy(s, REF, noise_floor=0.0)
        a1, _, _ = legacy(s, REF, noise_floor=1.0)
        self.assertLess(float(np.abs(a1).max()), float(np.abs(a0).max()))


if __name__ == "__main__":
    unittest.main()
