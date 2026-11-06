import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
KEYS = ("score", "no_at_fault_collisions", "drivable_area_compliance",
        "time_to_collision_within_bound", "ego_progress", "driving_direction_compliance")


def write_arm(root, name, rows):
    d = root / name / "validation"
    d.mkdir(parents=True)
    (d / "scores.json").write_text(json.dumps({"rows": rows}))


def rows(n, bump=0.0, fail_idx=()):
    out = []
    for i in range(n):
        r = {k: 1.0 for k in KEYS}
        r["token"] = f"t{i}"
        r["ego_progress"] = 0.7
        r["score"] = 0.9 + bump
        if i in fail_idx:
            r["drivable_area_compliance"] = 0.0
            r["score"] = 0.0
        out.append(r)
    return out


class SummaryTest(unittest.TestCase):
    def run_summary(self, root):
        res = subprocess.run([sys.executable, str(HERE / "summarize_offline_scaled.py"), str(root)],
                             capture_output=True, text=True)
        return res

    def test_pairing_conversions_and_smoke_ignored(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_arm(root, "scaled_v1_baseline_seed20", rows(20, fail_idx=(0, 1)))
            for seed in (20, 21):
                write_arm(root, f"scaled_v1_conditioned_seed{seed}", rows(20, bump=.02, fail_idx=(1, 5)))
            write_arm(root, "smoke_conditioned", rows(2))  # different scene set, must be ignored
            res = self.run_summary(root)
            self.assertEqual(res.returncode, 0, res.stderr)
            rep = json.loads((root / "scaled_comparison.json").read_text())
        conv = rep["vs_baseline"]["conditioned"]["failure_conversions"]["drivable_area_compliance"]
        self.assertEqual(conv, {"fixed": 1, "new_failures": 1})
        self.assertEqual(rep["scenes"], 20)
        self.assertEqual(sorted(rep["arms"]["conditioned"]["seeds"]), [20, 21])

    def test_counts_are_per_seed_then_averaged(self):
        from summarize_offline_scaled import compare
        def mapping(rs): return {r['token']:r for r in rs}
        base = mapping(rows(4, fail_idx=(0,)))
        a = mapping(rows(4))
        b = mapping(rows(4, fail_idx=(0, 1)))
        report = compare([a,b],[base],list(base),np.random.default_rng(0))
        self.assertEqual(report['failure_conversions']['drivable_area_compliance'],
                         {'fixed':.5, 'new_failures':.5})
        self.assertEqual(report['joint_safety']['fixed'], .5)

    def test_required_seeds_missing_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_arm(root,'scaled_v1_baseline_seed20', rows(4))
            write_arm(root,'scaled_v1_awfme0p3_seed20', rows(4))
            res = subprocess.run([sys.executable,str(HERE/'summarize_offline_scaled.py'),str(root),
                                  '--require-arm','awfme0p3:20,21'], capture_output=True,text=True)
            self.assertNotEqual(res.returncode,0)

    def test_scene_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_arm(root, "scaled_v1_baseline_seed20", rows(10))
            write_arm(root, "scaled_v1_conditioned_seed20", rows(9))
            self.assertNotEqual(self.run_summary(root).returncode, 0)


if __name__ == "__main__":
    unittest.main()
