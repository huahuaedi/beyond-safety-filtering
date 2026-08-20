import unittest

import numpy as np

from policy_optimization import (
    shrinkage_centered_advantage,
    update_adaptive_kl_coefficient,
)


class ShrinkageAdvantageTest(unittest.TestCase):
    def test_zero_floor_matches_group_zscore(self):
        values = np.array([0.1, 0.2, 0.4], dtype=np.float32)
        expected = (values - values.mean()) / (values.std() + 1e-6)
        np.testing.assert_allclose(
            shrinkage_centered_advantage(values), expected, rtol=1e-6, atol=1e-6
        )

    def test_floor_suppresses_tiny_differences(self):
        values = np.array([0.500, 0.501, 0.502], dtype=np.float32)
        legacy = shrinkage_centered_advantage(values)
        shrunk = shrinkage_centered_advantage(values, noise_floor=0.02)
        self.assertLess(shrunk.std(), legacy.std() * 0.1)
        self.assertAlmostEqual(float(shrunk.mean()), 0.0, places=6)

    def test_large_signal_is_mostly_preserved(self):
        values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        legacy = shrinkage_centered_advantage(values)
        shrunk = shrinkage_centered_advantage(values, noise_floor=0.02)
        self.assertGreater(shrunk.std(), legacy.std() * 0.99)


class AdaptiveKLTest(unittest.TestCase):
    def test_above_target_increases_coefficient(self):
        value = update_adaptive_kl_coefficient(
            0.5, 0.02, target_kl=0.01, adaptation_rate=0.2, minimum=0.1, maximum=2.0
        )
        self.assertGreater(value, 0.5)

    def test_below_target_decreases_coefficient(self):
        value = update_adaptive_kl_coefficient(
            0.5, 0.0, target_kl=0.01, adaptation_rate=0.2, minimum=0.1, maximum=2.0
        )
        self.assertLess(value, 0.5)

    def test_bounds_are_enforced(self):
        high = update_adaptive_kl_coefficient(
            1.9, 1.0, target_kl=0.01, adaptation_rate=2.0, minimum=0.1, maximum=2.0
        )
        low = update_adaptive_kl_coefficient(
            0.11, 0.0, target_kl=0.01, adaptation_rate=2.0, minimum=0.1, maximum=2.0
        )
        self.assertEqual(high, 2.0)
        self.assertEqual(low, 0.1)


if __name__ == "__main__":
    unittest.main()
