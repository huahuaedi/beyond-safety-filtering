import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
HEAD = ["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
        "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"]


def pdms(nc, dac, ep, ttc, c):
    return nc * dac * (5 * ep + 5 * ttc + 2 * c) / 12


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        for i, r in enumerate(rows):
            w.writerow([i, r["token"], "True", r["nc"], r["dac"], r["ep"], r["ttc"], 1.0, 1.0,
                        pdms(r["nc"], r["dac"], r["ep"], r["ttc"], 1.0)])
        w.writerow(["", "average", "", "", "", "", "", "", "", 0.5])   # must be ignored


def scenes(n, rng, ep_shift=0.0, ttc_fail=()):
    out = []
    for i in range(n):
        out.append({"token": f"t{i}", "nc": 1.0, "dac": 1.0, "ep": float(np.clip(0.7 + 0.1 * rng.normal() + ep_shift, 0, 1)),
                    "ttc": 0.0 if i in ttc_fail else 1.0})
    return out


class Original286Test(unittest.TestCase):
    def run_script(self, sft, rls, extra=()):
        out = sft.parent / "o.json"
        res = subprocess.run([sys.executable, str(HERE / "analyze_original_286.py"), "--sft", str(sft),
                              "--rl", *map(str, rls), "--output", str(out), *extra],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        return json.loads(out.read_text())

    def test_progress_gain_with_ttc_losses(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            base = scenes(60, rng)
            write_csv(t / "sft.csv", base)
            rl_paths = []
            for s, fails in enumerate(((1, 2), (3,), (4, 5))):
                rows = [dict(r, ep=min(1.0, r["ep"] + 0.03)) for r in base]
                for i in fails:
                    rows[i]["ttc"] = 0.0
                write_csv(t / f"rl{s}.csv", rows)
                rl_paths.append(t / f"rl{s}.csv")
            rep = self.run_script(t / "sft.csv", rl_paths)
        self.assertEqual(rep["scenes"], 60)
        self.assertGreater(rep["paired"]["ego_progress"]["delta_x100"], 2.0)
        self.assertEqual(rep["failure_conversions"]["time_to_collision_within_bound"]["new_failures_per_seed"], [2, 1, 2])
        self.assertEqual(rep["failure_conversions"]["time_to_collision_within_bound"]["fixed_per_seed"], [0, 0, 0])
        # average row must not leak in
        self.assertNotEqual(rep["scenes"], 61)

    def test_repeatability_and_identical_runs(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            base = scenes(30, rng)
            write_csv(t / "sft.csv", base)
            write_csv(t / "a.csv", base)
            write_csv(t / "a2.csv", base)
            rep = self.run_script(t / "sft.csv", [t / "a.csv"], ["--repeat", str(t / "a2.csv")])
        self.assertAlmostEqual(rep["paired"]["score"]["delta_x100"], 0.0)
        self.assertEqual(rep["repeatability"]["scenes_with_different_score"], [0])

    def test_log_cluster_interval_runs(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            base = scenes(24, rng)
            write_csv(t / "sft.csv", base)
            write_csv(t / "a.csv", [dict(r, ep=min(1, r["ep"] + 0.05)) for r in base])
            cache = t / "cache"
            for i in range(24):
                d = cache / f"log{i % 6}" / "unknown" / f"t{i}"
                d.mkdir(parents=True)
                (d / "metric_cache.pkl").touch()
            rep = self.run_script(t / "sft.csv", [t / "a.csv"], ["--cache-dir", str(cache)])
        self.assertEqual(rep["logs"], 6)
        lo, hi = rep["paired"]["score"]["log_cluster_ci95_x100"]
        self.assertGreater(lo, 0)
        self.assertLess(lo, hi)


if __name__ == "__main__":
    unittest.main()
