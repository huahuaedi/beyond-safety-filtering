import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress", "comfort")


def write(root, name, dac):
    d = {f"score__{k}": np.ones(4) for k in KEYS}
    d["score__drivable_area_compliance"] = np.asarray(dac, float)
    p = Path(root) / "collection" / "test" / f"{name}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **d)


class GuidedRepairTest(unittest.TestCase):
    def test_repaired_and_lost_scenes_are_counted(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write(t / "c", "a", [0, 0, 0, 0]); write(t / "g", "a", [0, 1, 0, 0])      # repaired
            write(t / "c", "b", [1, 0, 0, 0]); write(t / "g", "b", [0, 0, 0, 0])      # lost
            write(t / "c", "c", [0, 0, 0, 0]); write(t / "g", "c", [0, 0, 0, 0])      # unchanged
            res = subprocess.run([sys.executable, str(HERE / "analyze_guided_repair.py"), "--runs", f"lam0={t/'c'}", f"lam1={t/'g'}"],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            r = json.loads(res.stdout)["runs"]["lam1"]["drivable_area_compliance"]
        self.assertEqual((r["repaired_vs_control"], r["lost_vs_control"], r["scenes_with_a_passing_candidate"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
