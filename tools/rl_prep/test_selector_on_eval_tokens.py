import json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
from test_analyze_trajectory_selector import make_geom_scene

HERE = Path(__file__).parent


class EvalTokensTest(unittest.TestCase):
    def test_selector_evaluated_on_manifest_eval_tokens_only(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for split in ("train", "val", "test"):
                (t / "d" / split).mkdir(parents=True); (t / "g" / split).mkdir(parents=True)
            names = []
            for i in range(70):
                d, geom = make_geom_scene(rng)
                split = ("train", "val", "test")[i % 3]
                np.savez(t / "d" / split / f"s{i}.npz", **d)
                np.savez(t / "g" / split / f"s{i}.npz", geom_features=geom, feature_names=np.array(["x"] * 12))
                names.append((f"s{i}", split))
            train = [{"token": n, "split": s} for n, s in names[:50]]
            eval_tokens = [n for n, _ in names[50:]]
            (t / "m.json").write_text(json.dumps({"schema_version": 2, "train": train, "heldout_eval_tokens": eval_tokens}))
            out = t / "o.json"
            res = subprocess.run([sys.executable, str(HERE / "selector_on_eval_tokens.py"), "--root", str(t / "d"), "--geom-root", str(t / "g"),
                                  "--manifest", str(t / "m.json"), "--n-eval", "15", "--hgb-iters", "20", "--output", str(out)],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            r = json.loads(out.read_text())
        self.assertEqual(r["eval_scenes"], 15)                      # only the first 15 evaluation tokens
        self.assertEqual(r["train_scenes"], 50)
        self.assertGreater(r["models"]["hgb"]["gain_vs_reference_probes_x100"], r["models"]["hgb"]["random_gain_x100"])

    def test_overlap_between_train_and_eval_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for split in ("train", "val", "test"):
                (t / "d" / split).mkdir(parents=True); (t / "g" / split).mkdir(parents=True)
            d, geom = make_geom_scene(np.random.default_rng(0))
            np.savez(t / "d" / "train" / "a.npz", **d)
            np.savez(t / "g" / "train" / "a.npz", geom_features=geom, feature_names=np.array(["x"] * 12))
            (t / "m.json").write_text(json.dumps({"schema_version": 2, "train": [{"token": "a", "split": "train"}], "heldout_eval_tokens": ["a"]}))
            res = subprocess.run([sys.executable, str(HERE / "selector_on_eval_tokens.py"), "--root", str(t / "d"), "--geom-root", str(t / "g"),
                                  "--manifest", str(t / "m.json"), "--output", str(t / "o.json")], capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("overlap", res.stderr)


if __name__ == "__main__":
    unittest.main()
