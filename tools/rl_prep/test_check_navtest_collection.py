import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SCORES = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
          "ego_progress", "comfort", "driving_direction_compliance")


def write(path, progress=0.7, n=4):
    d = {"relative_trajectories": np.zeros((n, 8, 3)), "absolute_trajectories": np.zeros((n, 8, 3)),
         "expert_action_normalized": np.zeros((8, 3)), "reference_probe_absolute_trajectories": np.zeros((2, 8, 3))}
    for m in SCORES:
        v = progress if m == "ego_progress" else 1.0
        d["score__" + m] = np.full(n, v)
        d["reference_probe__" + m] = np.full(2, v)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)


def run(root, tokens=None):
    cmd = [sys.executable, str(HERE / "check_navtest_collection.py"), str(root)]
    if tokens:
        cmd.append(str(tokens))
    return subprocess.run(cmd, capture_output=True, text=True)


class CheckCollectionTest(unittest.TestCase):
    def test_good_collection_passes_and_stray_scene_fails(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write(t / "c" / "test" / "a.npz")
            write(t / "c" / "train" / "b.npz")
            (t / "tok.json").write_text(json.dumps(["a", "b"]))
            self.assertEqual(run(t / "c", t / "tok.json").returncode, 0)
            (t / "tok.json").write_text(json.dumps(["a"]))
            res = run(t / "c", t / "tok.json")
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("outside", res.stdout + res.stderr)

    def test_failed_scoring_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write(t / "c" / "test" / "a.npz", progress=0.0)
            res = run(t / "c")
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("scoring probably failed", res.stdout)


    def test_isolated_hard_scene_is_tolerated_but_majority_is_not(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for i in range(3):
                write(t / "c" / "train" / f"ok{i}.npz")
            write(t / "c" / "train" / "hard.npz", progress=0.0)
            res = run(t / "c")
            self.assertEqual(res.returncode, 0, res.stdout)
            self.assertIn("1 of 4 scenes have no progress signal", res.stdout)
            for i in range(3):
                write(t / "c" / "val" / f"hard{i}.npz", progress=0.0)
            self.assertNotEqual(run(t / "c").returncode, 0)


if __name__ == "__main__":
    unittest.main()
