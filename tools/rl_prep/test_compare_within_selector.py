import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent


def rows(vals):
    return [{"token": f"t{i}", "log": f"L{i % 3}", "pick": 0, "pdms_x100": v} for i, v in enumerate(vals)]


class CompareWithinTest(unittest.TestCase):
    def test_model_versus_ablation(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "r.json"
            f.write_text(json.dumps({"models": {"hgb": {"per_scene": rows([10, 20, 30, 40, 50, 60])}},
                                     "ablations": {"hgb_geom_only": {"per_scene": rows([10, 15, 30, 35, 50, 60])}}}))
            res = subprocess.run([sys.executable, str(HERE / "compare_within_selector.py"), str(f), "ablations.hgb_geom_only:models.hgb"],
                                 capture_output=True, text=True, cwd=HERE)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("models.hgb - ablations.hgb_geom_only = +1.667", res.stdout)


if __name__ == "__main__":
    unittest.main()
