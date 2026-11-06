import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MET = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
       "ego_progress", "comfort", "driving_direction_compliance")


class ProbeVsEvalTest(unittest.TestCase):
    def test_uniform_progress_shift_is_found(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "c" / "test").mkdir(parents=True)
            with open(t / "e.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["", "token", "valid", *MET, "score"])
                for i in range(40):
                    base = float(np.clip(0.6 + 0.1 * rng.normal(), 0, 0.9))
                    d = {k: np.ones(2) for k in MET}
                    d["ego_progress"] = np.array([base, base])           # probes: lower progress
                    z = {"reference_probe__" + k: v for k, v in d.items()}
                    np.savez(t / "c" / "test" / f"s{i}.npz", **z)
                    ep = base + 0.10                                      # the evaluation: +0.10 progress everywhere
                    sc = (5 * ep + 5 * 1 + 2 * 1) / 12
                    w.writerow([i, f"s{i}", "True", 1, 1, 1, ep, 1, 1, sc])
                w.writerow(["", "average", "", "", "", "", "", "", "", 0.5])
            res = subprocess.run([sys.executable, str(HERE / "analyze_probe_vs_eval.py"), "--collection", str(t / "c"),
                                  "--eval-csv", str(t / "e.csv"), "--output", str(t / "o.json")], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            r = json.loads((t / "o.json").read_text())
        self.assertEqual(r["scenes"], 40)
        self.assertAlmostEqual(r["per_metric_eval_minus_probe_x100"]["ego_progress"]["diff"], 10.0, places=4)
        self.assertAlmostEqual(r["per_metric_eval_minus_probe_x100"]["time_to_collision_within_bound"]["diff"], 0.0, places=6)
        self.assertEqual(r["concentration"]["share_scenes_identical"], 0.0)
        self.assertGreater(r["pdms_x100"]["eval_minus_probe"], 3.0)


if __name__ == "__main__":
    unittest.main()
