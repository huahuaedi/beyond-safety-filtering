import unittest

import numpy as np

from trajectory_dynamics import DynamicsLimits, analyze_pose_trajectories


class TrajectoryDynamicsTest(unittest.TestCase):
    def test_straight_constant_speed_is_feasible(self):
        x = np.arange(1, 9, dtype=np.float32) * 0.5
        trajectory = np.stack([x, np.zeros(8), np.zeros(8)], axis=1)[None]
        result = analyze_pose_trajectories(trajectory, 0.5)
        self.assertTrue(result["dynamics_feasible"][0])
        np.testing.assert_allclose(result["speed"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["curvature"], 0.0, atol=1e-6)

    def test_heading_jump_is_infeasible(self):
        x = np.arange(1, 9, dtype=np.float32) * 0.5
        heading = np.zeros(8, dtype=np.float32)
        heading[3:] = 1.0
        trajectory = np.stack([x, np.zeros(8), heading], axis=1)[None]
        result = analyze_pose_trajectories(trajectory, 0.5)
        self.assertFalse(result["dynamics_feasible"][0])
        self.assertGreater(result["max_abs_curvature"][0], 0.25)

    def test_acceleration_limit_is_detected(self):
        trajectory = np.zeros((1, 8, 3), dtype=np.float32)
        trajectory[0, :, 0] = np.asarray([0.1, 0.2, 0.3, 3, 6, 9, 12, 15])
        result = analyze_pose_trajectories(
            trajectory, 0.5, DynamicsLimits(max_speed=100.0)
        )
        self.assertFalse(result["dynamics_feasible"][0])
        self.assertGreater(result["max_acceleration"][0], 4.0)


if __name__ == "__main__":
    unittest.main()
