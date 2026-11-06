import unittest

import numpy as np

try:
    import torch
    import guidance_geometry as G
except ImportError:                                   # torch is only on the HPC and in the scratch environment
    torch = None

import geometric_features as gf

EGO = {"length": 4.6, "width": 2.0, "center_offset": 1.4}
TIMES = gf.HORIZON_TIMES


def rel_straight(step):
    rel = np.zeros((8, 3))
    rel[:, 0] = step
    return rel


def numpy_absolute(rel):
    x = y = yaw = 0.0
    out = np.zeros_like(rel)
    for i, (dx, dy, dyaw) in enumerate(rel):
        x += np.cos(yaw) * dx - np.sin(yaw) * dy
        y += np.sin(yaw) * dx + np.cos(yaw) * dy
        yaw += dyaw
        out[i] = [x, y, yaw]
    return out


def corridor_sdf(half_width=3.5, res=0.25, x0=-10.0, x1=60.0, y0=-20.0, y1=20.0):
    xs = np.arange(x0, x1 + 1e-9, res)
    ys = np.arange(y0, y1 + 1e-9, res)
    sdf = np.tile((half_width - np.abs(ys))[:, None], (1, len(xs)))                 # positive inside |y| < half_width
    return torch.tensor(sdf, dtype=torch.float32), (x0, y0), res


@unittest.skipIf(torch is None, "torch not installed")
class GuidanceGeometryTest(unittest.TestCase):
    def test_to_absolute_matches_numpy(self):
        rng = np.random.default_rng(0)
        rel = np.concatenate([rng.uniform(0.5, 4.0, (8, 1)), rng.normal(0, 0.3, (8, 1)), rng.normal(0, 0.08, (8, 1))], axis=1)
        got = G.to_absolute(torch.tensor(rel[None], dtype=torch.float64))[0].numpy()
        self.assertTrue(np.allclose(got, numpy_absolute(rel), atol=1e-9))

    def test_clearance_matches_numpy_features(self):
        rng = np.random.default_rng(1)
        rel = np.stack([rel_straight(3.0), rel_straight(2.0) + np.array([0, 0.3, 0.02])], axis=0)
        poses = np.stack([numpy_absolute(r) for r in rel])
        agents = {"pos": np.array([[14.0, 1.0], [30.0, -4.0]]), "vel": np.array([[1.0, 0.0], [-2.0, 0.5]]),
                  "heading": np.array([0.1, 3.0]), "length": np.array([4.0, 5.0]), "width": np.array([2.0, 2.1])}
        want, _ = gf.agent_clearance(poses, agents, EGO, TIMES)
        t_agents = {k: torch.tensor(v, dtype=torch.float64) for k, v in agents.items()}
        got = G.agent_clearance(torch.tensor(poses), t_agents, EGO, torch.tensor(TIMES, dtype=torch.float64)).numpy()
        self.assertTrue(np.allclose(got, want, atol=1e-6), (got, want))

    def test_cost_is_zero_for_safe_plan_and_positive_for_violations(self):
        sdf, origin, res = corridor_sdf()
        low, high = torch.tensor([-1.0, -3.0, -0.3]), torch.tensor([6.0, 3.0, 0.3])
        agents = {k: torch.tensor(v, dtype=torch.float32) for k, v in {
            "pos": np.array([[16.0, 0.0]]), "vel": np.zeros((1, 2)), "heading": np.zeros(1),
            "length": np.array([4.0]), "width": np.array([2.0])}.items()}
        cost = G.GuidanceCost(agents, sdf, origin, res, EGO, low, high, torch.tensor(TIMES, dtype=torch.float32))
        norm = lambda rel: 2 * (torch.tensor(rel, dtype=torch.float32) - low) / (high - low) - 1
        stop = norm(np.tile([0.2, 0.0, 0.0], (8, 1)))[None]                        # barely moves: far from the agent
        drive = norm(rel_straight(3.0))[None]                                    # drives into it
        swerve = norm(np.stack([np.array([3.0, 0.9, 0.0])] * 8))[None]           # drifts off the corridor
        self.assertEqual(float(cost(stop)), 0.0)
        self.assertGreater(float(cost(drive)), 0.0)
        self.assertGreater(float(cost(swerve)), 0.0)

    def test_guidance_reduces_violation_and_leaves_safe_plans_untouched(self):
        sdf, origin, res = corridor_sdf(half_width=8.0)                            # wide enough to go around the obstacle
        low, high = torch.tensor([-1.0, -3.0, -0.3]), torch.tensor([6.0, 3.0, 0.3])
        # the obstacle sits slightly to the left of the ego path; an exactly head-on obstacle has a zero lateral gradient
        agents = {k: torch.tensor(v, dtype=torch.float32) for k, v in {
            "pos": np.array([[16.0, 0.6]]), "vel": np.zeros((1, 2)), "heading": np.zeros(1),
            "length": np.array([4.0]), "width": np.array([2.0])}.items()}
        cost = G.GuidanceCost(agents, sdf, origin, res, EGO, low, high, torch.tensor(TIMES, dtype=torch.float32))
        norm = lambda rel: 2 * (torch.tensor(rel, dtype=torch.float32) - low) / (high - low) - 1

        def sample(target_rel, lam, seed=0):
            target = norm(target_rel)[None]
            torch.manual_seed(seed)
            z0 = torch.randn(1, 8, 3)
            vel = lambda z, t: (z - target) / max(t, 1e-3)                        # exact flow towards the target
            return G.guided_euler(vel, z0, 10, 0.0, cost_fn=cost, lam=lam, power=1.0)

        drive = rel_straight(3.0)
        plain, guided = sample(drive, 0.0), sample(drive, 0.4)
        self.assertLess(float(cost(guided)), float(cost(plain)))
        coll_p, road_p, clear_p, _ = cost.terms(plain)
        coll_g, road_g, clear_g, _ = cost.terms(guided)
        self.assertGreater(float(clear_g), float(clear_p))                        # larger minimum clearance
        self.assertEqual(float(road_g), 0.0)                                      # without leaving the drivable area
        safe = np.tile([0.2, 0.0, 0.0], (8, 1))
        self.assertTrue(torch.allclose(sample(safe, 0.0), sample(safe, 0.4), atol=1e-6))

    def test_gradient_is_clipped(self):
        called = {}

        def cost_fn(x):
            return 1e6 * (x ** 2).sum(dim=(1, 2))

        vel = lambda z, t: torch.zeros_like(z)
        z = G.guided_euler(vel, torch.ones(1, 8, 3), 10, 0.0, cost_fn=cost_fn, lam=0.1, max_grad_norm=1.0)
        self.assertTrue(torch.isfinite(z).all())
        self.assertLess(float((z - 1.0).abs().max()), 1.0)


if __name__ == "__main__":
    unittest.main()
