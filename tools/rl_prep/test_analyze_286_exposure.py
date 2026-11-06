import csv, json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
HEAD = ["", "token", "valid", "score", "ego_progress", "time_to_collision_within_bound", "drivable_area_compliance"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEAD)
        for i, (t, sc) in enumerate(rows):
            w.writerow([i, t, "True", sc, 0.7, 1.0, 1.0])
        w.writerow(["", "average", "", 0, 0, 0, 0])


class ExposureTest(unittest.TestCase):
    def test_exposed_scenes_show_the_planted_gain(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            pool, ev = t / "pool", t / "eval"
            # 20 logs; eval scene e{i} lives in log L{i}; pool scene p{i} also lives in L{i}
            for i in range(20):
                for base, name in ((pool, f"p{i}"), (ev, f"e{i}")):
                    d = base / f"L{i}" / "unknown" / name; d.mkdir(parents=True); (d / "metric_cache.pkl").touch()
            # the single seed trained on p0..p9 (steps 0..9) -> logs L0..L9 are exposed
            log = t / "train.log"
            log.write_text("".join(f"step {k:03d} scene=" + f"{k:016x} safe_frac=1\n" for k in range(0)) )
            # tokens in logs must be hex, so use hex-named pool scenes
            pool2 = t / "pool2"
            for i in range(20):
                for base, name in ((pool2, f"{i:016x}"), (ev, f"e{i}")):
                    d = base / f"L{i}" / "unknown" / name
                    d.mkdir(parents=True, exist_ok=True); (d / "metric_cache.pkl").touch()
            log.write_text("".join(f"step {k:03d} scene={k:016x} safe_frac=1.0000\n" for k in range(10)))
            sft_rows = [(f"e{i}", 0.80) for i in range(20)]
            rl_rows = [(f"e{i}", 0.80 + (0.05 if i < 10 else 0.0)) for i in range(20)]
            write_csv(t / "sft.csv", sft_rows); write_csv(t / "rl.csv", rl_rows)
            out = t / "o.json"
            res = subprocess.run([sys.executable, str(HERE / "analyze_286_exposure.py"), "--sft", str(t / "sft.csv"),
                                  "--rl", str(t / "rl.csv"), "--train-logs", str(log), "--steps", "10",
                                  "--pool-cache", str(pool2), "--eval-cache", str(ev), "--output", str(out)],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            r = json.loads(out.read_text())
        m = r["metrics"]["score"]
        self.assertEqual(m["pairs_exposed"], 10)
        self.assertAlmostEqual(m["mean_delta_exposed_x100"], 5.0, places=3)
        self.assertAlmostEqual(m["mean_delta_unexposed_x100"], 0.0, places=3)
        self.assertAlmostEqual(m["exposed_minus_unexposed_x100"], 5.0, places=3)
        self.assertGreater(m["log_bootstrap_ci95_x100"][0], 0)
        self.assertEqual(r["by_number_of_exposing_seeds"]["1"]["scenes"], 10)

    def test_steps_limit_the_trained_scenes(self):
        with tempfile.TemporaryDirectory() as t:
            log = Path(t) / "l.log"
            log.write_text("".join(f"step {k:03d} scene={k:016x} x=1\n" for k in range(30)))
            import analyze_286_exposure as a
            self.assertEqual(len(a.trained_tokens(log, 10)), 10)
            self.assertEqual(len(a.trained_tokens(log, 100)), 30)


if __name__ == "__main__":
    unittest.main()
