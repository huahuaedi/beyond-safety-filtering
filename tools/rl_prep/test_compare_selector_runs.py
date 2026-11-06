import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent


def run(path, gains):
    rows = [{"token": f"t{i}", "log": f"L{i % 3}", "pick": 0, "pdms_x100": g, "reference_probe_pdms_x100": 0.0} for i, g in enumerate(gains)]
    path.write_text(json.dumps({"models": {"ridge": {"per_scene": rows}, "hgb": {"per_scene": rows}}}))


class CompareTest(unittest.TestCase):
    def test_paired_difference_and_counts(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            run(t / "a.json", [10, 20, 30, 40, 50, 60])
            run(t / "b.json", [10, 25, 30, 45, 55, 55])
            res = subprocess.run([sys.executable, str(HERE / "compare_selector_runs.py"), str(t / "a.json"), str(t / "b.json")],
                                 capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("hgb: B - A = +1.667", res.stdout)
        self.assertIn("B better in 3, A better in 1", res.stdout)


if __name__ == "__main__":
    unittest.main()
