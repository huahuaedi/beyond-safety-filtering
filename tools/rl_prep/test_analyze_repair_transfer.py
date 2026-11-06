import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_repair_transfer as A
from offline_verified_data import absolute_from_relative, SAFETY


def write_scene(root, token, n_pass, n=16):
    rel = (np.random.default_rng(abs(hash(token)) % 1000).normal(size=(n, 8, 3)) * 0.1).astype(np.float32)
    d = Path(root) / "train"
    d.mkdir(parents=True, exist_ok=True)
    nc = np.zeros(n, np.float32)
    nc[:n_pass] = 1
    payload = {"scene_token": np.array(token), "split": np.array("train"), "relative_trajectories": rel,
               "absolute_trajectories": absolute_from_relative(rel), "reference_probe__ego_progress": np.array([0.5, 0.5], np.float32)}
    for k in SAFETY:
        payload["score__" + k] = nc if k == "no_at_fault_collisions" else np.ones(n, np.float32)
        payload["reference_probe__" + k] = np.ones(2, np.float32)
    payload["score__ego_progress"] = np.full(n, 0.5, np.float32)
    np.savez(d / f"{token}.npz", **payload)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "valid", "score", *A.CRIT])
        w.writeheader()
        for token, nc in rows.items():
            w.writerow({"token": token, "valid": "True", "score": 0.9, "no_at_fault_collisions": nc,
                        "drivable_area_compliance": 1, "time_to_collision_within_bound": 1})


class AnalyseTest(unittest.TestCase):
    def test_component_repair_does_not_imply_joint_repair(self):
        b = {'s': dict(zip(A.CRIT, [0., 0., 1.]))}
        a = {'s': dict(zip(A.CRIT, [1., 0., 1.]))}
        fixed, new = A.flips(b, a, ['s'])
        self.assertEqual((fixed[0], new[0]), (0., 0.))
        a['s'][A.CRIT[1]] = 1.
        fixed, new = A.flips(b, a, ['s'])
        self.assertEqual((fixed[0], new[0]), (1., 0.))

    def test_trading_failures_is_not_joint_repair_or_regression(self):
        b = {'s': dict(zip(A.CRIT, [0., 1., 1.]))}
        a = {'s': dict(zip(A.CRIT, [1., 0., 1.]))}
        f, n = A.flips(b, a, ['s'])
        self.assertEqual((f[0], n[0]), (0., 0.))

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for token, k in (("a", 0), ("b", 4), ("c", 16), ("d", 8)):
                write_scene(t / "pool", token, k)
            write_csv(t / "base.csv", {"a": 0, "b": 0, "c": 1, "d": 1})
            write_csv(t / "arm.csv", {"a": 0, "b": 1, "c": 0, "d": 1})
            res = A.analyse(t / "pool", t / "base.csv", {"arm": t / "arm.csv"}, [0.3])
            self.assertAlmostEqual(res["share_with_failing_candidate"], 0.75)
            self.assertAlmostEqual(res["share_without_any_passer"], 0.25)
            arm = res["arms"]["arm"]
            self.assertEqual((arm["fixed"], arm["new"]), (1.0, 1.0))
            rows = {(r["p_lo"], r["p_hi"]): r for r in arm["by_pool_pass_rate"]["0.3"]}
            self.assertEqual(rows[(0.0, 0.02)]["n"], 1)          # scene a: p = 0
            self.assertAlmostEqual(rows[(0.0, 0.02)]["target_pool_pass_gain"], 0.0)
            self.assertAlmostEqual(rows[(0.1, 0.3)]["fixed"], 1.0)   # scene b: p = 0.25
            self.assertGreater(rows[(0.1, 0.3)]["target_pool_pass_gain"], 0.0)
            self.assertAlmostEqual(rows[(0.7, 1.0001)]["new"], 1.0)  # scene c: p = 1, regressed


if __name__ == "__main__":
    unittest.main()
