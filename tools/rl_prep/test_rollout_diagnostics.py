import unittest

import numpy as np

from rollout_diagnostics import compute_rollout_diagnostics


def scores(progress):
    size = len(progress)
    return {
        "ego_progress": np.asarray(progress, dtype=np.float32),
        "no_at_fault_collisions": np.ones(size, dtype=np.float32),
        "drivable_area_compliance": np.ones(size, dtype=np.float32),
        "time_to_collision_within_bound": np.ones(size, dtype=np.float32),
    }


class RolloutDiagnosticsTest(unittest.TestCase):
    def test_identical_group_is_zero_signal(self):
        trajs = np.zeros((4, 3, 3), dtype=np.float32)
        result = compute_rollout_diagnostics(
            trajs, scores([1, 1, 1, 1]), np.zeros(4), np.zeros(4), np.ones(4, bool)
        )
        self.assertEqual(result["trajectory_pairwise_rmse"], 0.0)
        self.assertEqual(result["safe_candidates"], 4)
        self.assertEqual(result["zero_signal_group"], 1)

    def test_trajectory_diversity_prevents_zero_signal(self):
        trajs = np.zeros((2, 2, 3), dtype=np.float32)
        trajs[1, :, 0] = 2.0
        result = compute_rollout_diagnostics(
            trajs, scores([1, 1]), np.zeros(2), np.zeros(2), np.ones(2, bool)
        )
        self.assertAlmostEqual(result["trajectory_pairwise_rmse"], np.sqrt(2.0))
        self.assertEqual(result["zero_signal_group"], 0)

    def test_raw_score_spread_is_recorded(self):
        trajs = np.zeros((2, 2, 3), dtype=np.float32)
        result = compute_rollout_diagnostics(
            trajs, scores([0, 1]), np.array([-1, 1]), np.zeros(2), np.ones(2, bool)
        )
        self.assertAlmostEqual(result["progress_raw_std"], 0.5)
        self.assertAlmostEqual(result["adv_effective_range"], 2.0)
        self.assertEqual(result["zero_signal_group"], 0)


if __name__ == "__main__":
    unittest.main()
