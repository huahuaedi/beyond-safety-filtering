import unittest

import numpy as np

from adaptive_reward import (
    estimate_tail_risk_baselines,
    scene_constraint_weights,
    validate_constraint_weights,
)


class AdaptiveRewardTest(unittest.TestCase):
    def test_equal_risk_produces_unit_weights(self):
        probabilities = np.full((16, 4), 0.2, dtype=np.float32)
        weights, tail = scene_constraint_weights(probabilities, topk=8)
        self.assertEqual(weights, {"nc": 1.0, "dac": 1.0, "ttc": 1.0})
        self.assertTrue(all(abs(value - 0.2) < 1e-6 for value in tail.values()))

    def test_dominant_ttc_increases_ttc_emphasis(self):
        probabilities = np.full((16, 4), 0.1, dtype=np.float32)
        probabilities[:, 1] = 0.9
        weights, _ = scene_constraint_weights(probabilities, topk=8)
        self.assertGreater(weights["ttc"], 1.0)
        self.assertLess(weights["nc"], 1.0)
        self.assertLess(weights["dac"], 1.0)

    def test_weights_are_bounded(self):
        probabilities = np.zeros((8, 4), dtype=np.float32)
        probabilities[:, 3] = 1.0
        weights, _ = scene_constraint_weights(
            probabilities, topk=8, strength=1.0, minimum=0.5, maximum=2.0
        )
        self.assertEqual(weights, {"nc": 2.0, "dac": 0.5, "ttc": 0.5})

    def test_head_scale_calibration_removes_systematic_bias(self):
        probabilities = np.zeros((10, 8, 4), dtype=np.float32)
        probabilities[:, :, 1] = 0.2
        probabilities[:, :, 2] = 0.8
        probabilities[:, :, 3] = 0.05
        baselines = estimate_tail_risk_baselines(probabilities, topk=4)
        weights, _ = scene_constraint_weights(
            probabilities[0], topk=4, risk_baselines=baselines
        )
        self.assertTrue(all(abs(value - 1.0) < 1e-6 for value in weights.values()))

    def test_validation_rejects_bad_weights(self):
        with self.assertRaises(KeyError):
            validate_constraint_weights({"nc": 1.0, "dac": 1.0})
        with self.assertRaises(ValueError):
            validate_constraint_weights(
                {"nc": 0.0, "dac": 1.0, "ttc": 1.0}, minimum=0.5
            )


if __name__ == "__main__":
    unittest.main()
