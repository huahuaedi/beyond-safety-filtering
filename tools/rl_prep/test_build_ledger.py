import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
HEAD = ["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "score"]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEAD)
        for i, (nc, dac, ttc, sc) in enumerate(rows):
            w.writerow([i, f"t{i}", "True", nc, dac, ttc, sc])
        w.writerow(["", "average", "", "", "", "", 0.5])


class LedgerTest(unittest.TestCase):
    def test_group_and_selector_rows(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write(t / "m" / "x_sft" / "a" / "b.csv", [(1, 0, 1, 0.5), (1, 1, 0, 0.6), (1, 1, 1, 0.9), (1, 1, 1, 0.9)])
            write(t / "m" / "x_arm_seed0" / "a" / "b.csv", [(1, 1, 1, 0.9), (1, 1, 0, 0.6), (1, 0, 1, 0.5), (0, 1, 1, 0.4)])
            sel = {"models": {"hgb": {"detail": {"vs_independent_sft": {"scenes": 4, "gain_x100": 1.5, "fix_break": {
                "no_at_fault_collisions": {"fixed": 1, "broken": 2}, "drivable_area_compliance": {"fixed": 3, "broken": 0},
                "time_to_collision_within_bound": {"fixed": 0, "broken": 4}}}}}}}
            (t / "s.json").write_text(json.dumps(sel))
            res = subprocess.run([sys.executable, str(HERE / "build_ledger.py"),
                                  "--group", f"RL arm|{t}/m/*_sft/*/*.csv|{t}/m/*_arm_seed*/*/*.csv", "--selector", f"Selector=" + f"{t}/s.json:hgb",
                                  "--output", str(t / "o.json")], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            rows = json.loads((t / "o.json").read_text())
        grp, selr = rows
        self.assertEqual(grp["fixed"]["drivable_area_compliance"], 1.0)      # scene 0 fixed
        self.assertEqual(grp["new"]["drivable_area_compliance"], 1.0)        # scene 2 broken
        self.assertEqual(grp["new"]["no_at_fault_collisions"], 1.0)
        self.assertAlmostEqual(selr["net_passing"], (1 - 2) + (3 - 0) + (0 - 4))


if __name__ == "__main__":
    unittest.main()
