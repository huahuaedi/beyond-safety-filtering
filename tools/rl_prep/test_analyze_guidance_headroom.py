import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound")


def write(path, dac, ref_dac):
    d = {f"score__{k}": np.ones(4) for k in KEYS}
    d.update({f"reference_probe__{k}": np.ones(2) for k in KEYS})
    d["score__drivable_area_compliance"] = np.asarray(dac, float)
    d["reference_probe__drivable_area_compliance"] = np.asarray(ref_dac, float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)


class HeadroomTest(unittest.TestCase):
    def test_repairable_and_unrepairable_scenes_are_separated(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write(t / "c" / "test" / "a.npz", [0, 1, 1, 1], [0, 0])      # reference fails, a candidate passes -> repairable
            write(t / "c" / "test" / "b.npz", [0, 0, 0, 0], [0, 1])      # reference fails, every candidate fails -> unrepairable
            write(t / "c" / "test" / "c.npz", [0, 0, 0, 0], [1, 1])      # reference passes, every candidate fails
            write(t / "c" / "test" / "d.npz", [1, 1, 1, 1], [1, 1])
            res = subprocess.run([sys.executable, str(HERE / "analyze_guidance_headroom.py"), "--collection", str(t / "c")],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            r = json.loads(res.stdout)["drivable_area_compliance"]
        self.assertEqual(r["ref_fail"], 2)
        self.assertEqual(r["ref_fail_repairable"], 1)
        self.assertEqual(r["ref_fail_unrepairable"], 1)
        self.assertEqual(r["ref_ok_all_cand_fail"], 1)
        self.assertEqual(r["scenes_all_cand_fail"], 2)
        self.assertAlmostEqual(r["candidate_failure_rate"], (0.25 + 1.0 + 1.0 + 0.0) / 4)


if __name__ == "__main__":
    unittest.main()
