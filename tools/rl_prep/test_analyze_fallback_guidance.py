import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def result(path, rows):
    path.write_text(json.dumps({"models": {"hgb": {"per_scene": [
        {"token": t, "log": f"L{i % 2}", "pick": p, "pdms_x100": v, "reference_probe_pdms_x100": 0} for i, (t, p, v) in enumerate(rows)]}}}))


class FallbackTest(unittest.TestCase):
    def test_only_violating_picks_are_replaced(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "geom" / "test").mkdir(parents=True)
            for token, bad_pick in (("a", True), ("b", False), ("c", True)):
                g = np.zeros((4, 12))
                g[0, 8] = 0.5 if bad_pick else 0.0                      # candidate 0 leaves the drivable area in a and c
                g[:, 0] = 1.0
                np.savez(t / "geom" / "test" / f"{token}.npz", geom_features=g)
            result(t / "u.json", [("a", 0, 0.0), ("b", 0, 90.0), ("c", 0, 0.0)])
            result(t / "g.json", [("a", 1, 80.0), ("b", 1, 10.0), ("c", 1, -0.0)])
            res = subprocess.run([sys.executable, str(HERE / "analyze_fallback_guidance.py"), "--unguided-result", str(t / "u.json"),
                                  "--guided-result", str(t / "g.json"), "--unguided-geom", str(t / "geom")], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("= +26.667", res.stdout)          # (80 + 0 + 0) / 3: b keeps its unguided 90 although the guided pick is worse
        self.assertIn("switched 2 scenes", res.stdout)


if __name__ == "__main__":
    unittest.main()
