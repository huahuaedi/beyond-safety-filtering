import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
KEYS = ("score", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress")


def write_run(root, name, rows):
    d = root / name / "validation"
    d.mkdir(parents=True)
    (d / "scores.json").write_text(json.dumps({"rows": rows}))
    return root / name


def rows(tokens, rng, progress_shift=0.0, ttc_fail=()):
    out = []
    for i, t in enumerate(tokens):
        ep = float(np.clip(0.7 + 0.05 * rng.normal() + progress_shift, 0, 1))
        ttc = 0.0 if i in ttc_fail else 1.0
        out.append({"token": t, "no_at_fault_collisions": 1.0, "drivable_area_compliance": 1.0,
                    "time_to_collision_within_bound": ttc, "ego_progress": ep, "score": (5 * ep + 5 * ttc + 2) / 12})
    return out


class SummarizePolicyEvalTest(unittest.TestCase):
    def test_drift_family_differs_from_clean_family(self):
        rng = np.random.default_rng(0)
        tokens = [f"t{i}" for i in range(60)]
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            runs = [f"baseline={write_run(t, 'baseline', rows(tokens, rng))}"]
            for seed in (20, 21):
                runs.append(f"driftpref_seed{seed}={write_run(t, f'd{seed}', rows(tokens, rng, 0.08, ttc_fail=(1, 2, 3, 4, 5)))}")
                runs.append(f"cleanpref_seed{seed}={write_run(t, f'c{seed}', rows(tokens, rng, 0.0))}")
            mapping = {tok: f"log_name:L{i % 10}" for i, tok in enumerate(tokens)}
            (t / "map.json").write_text(json.dumps(mapping))
            res = subprocess.run([sys.executable, str(HERE / "summarize_policy_eval.py"), "--runs", *runs,
                                  "--token-log-map", str(t / "map.json"), "--output", str(t / "o.json")], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            r = json.loads((t / "o.json").read_text())
        self.assertEqual(r["scenes"], 60)
        self.assertEqual(r["logs"], 10)
        self.assertEqual(set(r["families"]), {"driftpref", "cleanpref"})
        d, c = r["families"]["driftpref"], r["families"]["cleanpref"]
        self.assertGreater(d["vs_baseline"]["ego_progress"]["delta_x100"], 5.0)
        self.assertLess(d["vs_baseline"]["time_to_collision_within_bound"]["delta_x100"], -5.0)
        self.assertAlmostEqual(c["vs_baseline"]["time_to_collision_within_bound"]["delta_x100"], 0.0)
        self.assertEqual(d["fix_break_mean_over_seeds"]["time_to_collision_within_bound"]["new"], 5.0)
        pair = r["pairs"]["cleanpref minus driftpref"]
        self.assertLess(pair["ego_progress"]["delta_x100"], -5.0)
        lo, hi = pair["ego_progress"]["log_ci95_x100"]
        self.assertLess(hi, 0.0)


if __name__ == "__main__":
    unittest.main()
