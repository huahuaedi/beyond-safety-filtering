import csv, tempfile, unittest
from pathlib import Path
import analyze_fix_break as F

HEAD = ["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "score"]


def write(root, name, rows):
    d = Path(root) / "metrics" / name / "2026.01.01"; d.mkdir(parents=True)
    with open(d / "x.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEAD)
        for i, (t, nc, dac, ttc) in enumerate(rows):
            w.writerow([i, t, "True", nc, dac, ttc, 0.9])
        w.writerow(["", "average", "", 0, 0, 0, 0])


class FixBreakTest(unittest.TestCase):
    def test_counts_and_rates(self):
        with tempfile.TemporaryDirectory() as t:
            tokens = [f"t{i}" for i in range(10)]
            # SFT: TTC fails in t0, t1; all else passes
            write(t, "run_sft", [(x, 1, 1, 0 if x in ("t0", "t1") else 1) for x in tokens])
            # seed0: fixes t0, breaks t5 and t6
            write(t, "run_a_seed0", [(x, 1, 1, 0 if x in ("t1", "t5", "t6") else 1) for x in tokens])
            # seed1: fixes nothing, breaks t7
            write(t, "run_a_seed1", [(x, 1, 1, 0 if x in ("t0", "t1", "t7") else 1) for x in tokens])
            sft = F.arm_csv(t, "sft")
            ps = [F.transitions(sft, F.arm_csv(t, f"a_seed{s}"))[0] for s in range(2)]
            s = F.summarize(ps)["TTC"]
            self.assertEqual((s["sft_fail"], s["sft_pass"]), (2, 8))
            self.assertEqual(s["fixed_per_seed"], [1, 0]); self.assertEqual(s["broken_per_seed"], [2, 1])
            self.assertAlmostEqual(s["P_fix"], 0.25); self.assertAlmostEqual(s["P_break"], 1.5 / 8)
            self.assertAlmostEqual(s["net_passing_scenes"], -1.0)
            self.assertEqual(F.summarize(ps)["NC"]["fixed_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
