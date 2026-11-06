import unittest

import numpy as np

import geometric_features as gf

EGO = {"length": 4.6, "width": 2.0, "center_offset": 1.4}
NAMES = {name: i for i, name in enumerate(gf.FEATURE_NAMES)}
TIMES = gf.HORIZON_TIMES


def straight(speed, y=0.0):
    x = speed * TIMES
    return np.stack([x, np.full_like(x, y), np.zeros_like(x)], axis=-1)


def agents(pos, vel=(0.0, 0.0), heading=0.0, length=4.0, width=2.0):
    return {
        "pos": np.array([pos], dtype=np.float64),
        "vel": np.array([vel], dtype=np.float64),
        "heading": np.array([heading]),
        "length": np.array([length]),
        "width": np.array([width]),
    }


def always_inside(points):
    return np.ones(points.shape[:-1], dtype=bool)


def strip(half_width):
    return lambda points: np.abs(points[..., 1]) < half_width


def feats(candidates, scene_agents, inside=always_inside, origin=(0.0, 0.0, 0.0)):
    return gf.scene_geometric_features(np.stack(candidates), origin, scene_agents, EGO, inside)


class GeometricFeatureTest(unittest.TestCase):
    def test_no_agents_saturates_clearance_and_never_collides(self):
        f = feats([straight(5.0)], None)
        self.assertAlmostEqual(f[0, NAMES["min_clearance"]], gf.CLEARANCE_CAP / 10.0)
        self.assertEqual(f[0, NAMES["t_clear_lt_0m"]], 1.0)
        self.assertEqual(f[0, NAMES["n_agents_lt_3m"]], 0.0)
        self.assertTrue(np.isfinite(f).all())

    def test_stationary_obstacle_separates_driving_through_from_stopping(self):
        f = feats([straight(5.0), straight(0.0)], agents((12.0, 0.0)))
        drive, stop = f
        self.assertLess(drive[NAMES["min_clearance"]], 0.0)  # overlaps the obstacle
        self.assertLess(drive[NAMES["t_clear_lt_0m"]], 1.0)  # collision time found
        self.assertGreater(stop[NAMES["min_clearance"]], 0.0)
        self.assertEqual(stop[NAMES["t_clear_lt_0m"]], 1.0)  # never collides
        self.assertLess(drive[NAMES["min_headway"]], stop[NAMES["min_headway"]])

    def test_constant_velocity_forecast_finds_oncoming_agent(self):
        # Ego stands still; agent 25 m ahead closes at 8 m/s => contact ~ 2.5-3 s.
        f = feats([straight(0.0)], agents((25.0, 0.0), vel=(-8.0, 0.0)))
        t_collision = f[0, NAMES["t_clear_lt_0m"]] * gf.NO_EVENT_TIME
        self.assertGreaterEqual(t_collision, 1.5)
        self.assertLessEqual(t_collision, 3.5)
        # A static (velocity 0) forecast of the same agent must not collide.
        static = feats([straight(0.0)], agents((25.0, 0.0)))
        self.assertEqual(static[0, NAMES["t_clear_lt_0m"]], 1.0)

    def test_parallel_lateral_clearance_matches_analytic_value(self):
        # Centre lines 3.5 m apart, widths 2.0 + 2.0 -> 1.5 m clearance.
        f = feats([straight(0.0)], agents((EGO["center_offset"], 3.5)))
        self.assertAlmostEqual(f[0, NAMES["min_clearance"]] * 10.0, 1.5, places=3)

    def test_drivable_area_features(self):
        curve = np.stack([2.0 * TIMES, np.linspace(0.0, 9.0, 8), np.zeros(8)], axis=-1)
        f = feats([straight(3.0), curve], None, inside=strip(3.5))
        keep, leave = f
        self.assertEqual(keep[NAMES["offroad_frac_max"]], 0.0)
        self.assertEqual(keep[NAMES["t_first_offroad"]], 1.0)
        self.assertGreater(leave[NAMES["offroad_frac_max"]], 0.0)
        self.assertGreater(leave[NAMES["offroad_frac_final"]], 0.9)
        self.assertLess(leave[NAMES["t_first_offroad"]], 1.0)

    def test_features_are_invariant_to_the_global_frame(self):
        rng = np.random.default_rng(3)
        cands = [np.stack([3.0 * TIMES, rng.normal(0, 0.3, 8).cumsum(), np.zeros(8)], -1) for _ in range(6)]
        local_agents = agents((14.0, 1.0), vel=(-2.0, 0.5), heading=0.2)
        base = feats(cands, local_agents)

        origin = (100.0, -40.0, 1.1)
        c, s = np.cos(origin[2]), np.sin(origin[2])
        rot = np.array([[c, -s], [s, c]])
        moved = {
            "pos": local_agents["pos"] @ rot.T + np.array(origin[:2]),
            "vel": local_agents["vel"] @ rot.T,
            "heading": local_agents["heading"] + origin[2],
            "length": local_agents["length"],
            "width": local_agents["width"],
        }
        # A drivable strip defined in the local frame, expressed globally.
        def inside_global(points):
            local = (points - np.array(origin[:2])) @ rot
            return np.abs(local[..., 1]) < 2.0

        base = feats(cands, local_agents, inside=strip(2.0))
        moved_features = feats(cands, moved, inside=inside_global, origin=origin)
        np.testing.assert_allclose(base, moved_features, atol=1e-5)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            gf.scene_geometric_features(np.zeros((2, 5, 3)), (0, 0, 0), None, EGO, always_inside)
        bad = straight(1.0)[None].copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            gf.scene_geometric_features(bad, (0, 0, 0), None, EGO, always_inside)
        with self.assertRaises(ValueError):
            gf.scene_geometric_features(
                straight(1.0)[None], (0, 0, 0), None, EGO, lambda p: np.ones(p.shape[:-1] + (1,), bool)
            )


TTC = {name: i for i, name in enumerate(gf.TTC_FEATURE_NAMES)}


def ttc_feats(candidates, scene_agents):
    return gf.scene_ttc_features(np.stack(candidates), (0.0, 0.0, 0.0), scene_agents, EGO)


class TtcFeatureTest(unittest.TestCase):
    def test_slow_lead_agent_is_seen_between_grid_points(self):
        # Lead car 14 m ahead at 2 m/s. Ego at 8 m/s is clear on the 0.5 s grid at first,
        # but the 1 s constant-velocity projection reaches it.
        lead = agents((14.0, 0.0), vel=(2.0, 0.0))
        fast, stop = ttc_feats([straight(8.0), straight(0.0)], lead)
        self.assertGreater(fast[TTC["ttc_violation_frac"]], 0.0)
        self.assertLess(fast[TTC["ttc_first_violation"]], 1.0)
        self.assertEqual(stop[TTC["ttc_violation_frac"]], 0.0)       # stationary ego is exempt
        self.assertEqual(stop[TTC["ttc_first_violation"]], 1.0)
        self.assertLess(fast[TTC["ttc_min_proj_clearance"]], stop[TTC["ttc_min_proj_clearance"]])

    def test_no_agents_is_clean(self):
        f = ttc_feats([straight(6.0)], None)
        self.assertEqual(f[0, TTC["ttc_violation_frac"]], 0.0)
        self.assertTrue(np.isfinite(f).all())
        self.assertEqual(f.shape, (1, len(gf.TTC_FEATURE_NAMES)))

    def test_lateral_offset_avoids_violation(self):
        lead = agents((14.0, 0.0), vel=(2.0, 0.0))
        straight_on, swerve = ttc_feats([straight(8.0), straight(8.0, y=6.0)], lead)
        self.assertGreater(straight_on[TTC["ttc_violation_frac"]], swerve[TTC["ttc_violation_frac"]])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            gf.scene_ttc_features(np.zeros((2, 8)), (0, 0, 0), None, EGO)


if __name__ == "__main__":
    unittest.main()
