import unittest

import numpy as np

import paired_scores_by_log as P


class PairedByLogTest(unittest.TestCase):
    def test_mean_interval_and_win_loss_sizes(self):
        tokens = [f"t{i}" for i in range(40)]
        base = {t: {"score": 0.5, "ego_progress": 0.5} for t in tokens}
        arm = {t: {"score": 0.5 + (0.2 if i < 4 else -0.01 if i < 20 else 0.0), "ego_progress": 0.5} for i, t in enumerate(tokens)}
        mapping = {t: f"log_name:L{i % 8}" for i, t in enumerate(tokens)}
        r = P.compare(base, arm, mapping, n=500)
        s = r["metrics"]["score"]
        self.assertEqual((s["wins"], s["losses"], s["ties"]), (4, 16, 20))
        self.assertAlmostEqual(s["mean_x100"], (4 * 20 - 16 * 1) / 40)
        self.assertAlmostEqual(s["mean_win_x100"], 20.0); self.assertAlmostEqual(s["mean_loss_x100"], -1.0)
        lo, hi = s["log_ci95_x100"]
        self.assertLessEqual(lo, s["mean_x100"] + 1e-9); self.assertGreaterEqual(hi, s["mean_x100"] - 1e-9)

    def test_mean_of_tables(self):
        a = {"t0": {"score": 0.2}, "t1": {"score": 0.4}}; b = {"t0": {"score": 0.6}, "t1": {"score": 0.0}}
        import tempfile, csv
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, tab in enumerate((a, b)):
                p = Path(d) / f"{i}.csv"
                with open(p, "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["", "token", "valid", "score"])
                    for k, (tok, v) in enumerate(tab.items()):
                        w.writerow([k, tok, "True", v["score"]])
                paths.append(p)
            m = P.read_mean(paths)
        self.assertAlmostEqual(m["t0"]["score"], 0.4); self.assertAlmostEqual(m["t1"]["score"], 0.2)

    def test_scene_sets_must_match(self):
        with self.assertRaises(ValueError):
            P.compare({"a": {"score": 1.0}}, {"b": {"score": 1.0}}, {"a": "log_name:x", "b": "log_name:x"})


if __name__ == "__main__":
    unittest.main()
