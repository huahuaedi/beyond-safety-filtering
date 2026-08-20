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

    def test_friction_circle_combines_longitudinal_and_lateral_acceleration(self):
        # Neither component alone exceeds 4 m/s^2, but their vector norm does.
        dt = 0.5
        speed = np.asarray([2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5])
        distance = speed * dt
        heading_step = 0.04 * distance
        x = np.cumsum(distance)
        heading = np.cumsum(heading_step)
        trajectory = np.stack([x, np.zeros(8), heading], axis=1)[None]
        result = analyze_pose_trajectories(
            trajectory, dt,
            DynamicsLimits(
                max_speed=100.0, max_acceleration=100.0, max_abs_jerk=100.0,
                max_abs_curvature=1.0, max_abs_curvature_rate=1.0,
                friction_coefficient=0.40, gravity=9.81,
            ),
        )
        self.assertGreater(result["max_friction_acceleration"][0], 0.40 * 9.81)
        self.assertGreater(result["violation_components"][0, 6], 0.0)

    def test_per_candidate_friction_is_supported(self):
        x = np.arange(1, 9, dtype=np.float32) * 2.0
        heading = np.arange(1, 9, dtype=np.float32) * 0.08
        one = np.stack([x, np.zeros(8), heading], axis=1)
        result = analyze_pose_trajectories(
            np.stack([one, one]), 0.5,
            DynamicsLimits(
                max_speed=100.0, max_abs_curvature=1.0,
                max_abs_curvature_rate=1.0,
                friction_coefficient=np.asarray([0.05, 1.0]),
            ),
        )
        self.assertFalse(result["dynamics_feasible"][0])
        self.assertTrue(result["dynamics_feasible"][1])


if __name__ == "__main__":
    unittest.main()
