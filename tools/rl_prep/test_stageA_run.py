import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_analyze_trajectory_selector import make_geom_scene

HERE = Path(__file__).parent


def build(root, n_scenes, rng):
    """Pool scenes whose DAC failure is visible only in the (synthetic) tokens: token 0 carries the sign of the lateral offset."""
    toks, logs = [], {}
    for i in range(n_scenes):
        d, geom = make_geom_scene(rng, n=16)
        name = f"s{i}"
        split = ("train", "val", "test")[i % 3]
        for sub in ("c", "g"):
            (root / sub / split).mkdir(parents=True, exist_ok=True)
        np.savez(root / "c" / split / f"{name}.npz", **d)
        np.savez(root / "g" / split / f"{name}.npz", geom_features=geom, feature_names=np.array(["x"] * 12))
        toks.append(name); logs[name] = f"log_name:L{i % 12}"
    return toks, logs


def write_tokens(path, names, rng):
    path.mkdir(parents=True, exist_ok=True)
    grid = np.stack([np.zeros(36, int), np.arange(36) // 6, np.arange(36) % 6], axis=1).astype(np.int16)
    np.savez(path / "tokens_shard0.npz", tokens=np.array(names), grid=grid, h=rng.normal(size=(len(names), 36, 128)).astype(np.float16),
             projection_fingerprint=-0.3)


class RunTest(unittest.TestCase):
    def test_end_to_end_all_variants_and_compare(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            toks, logs = build(t, 90, rng)
            write_tokens(t / "tok", toks, rng)
            (t / "map.json").write_text(json.dumps(logs))
            train = toks[:50]
            (t / "m.json").write_text(json.dumps({"train": [{"token": x} for x in train]}))
            out = t / "out"
            for v in "abcde":
                for seed in (0, 1):
                    cmd = [sys.executable, str(HERE / "stageA_run.py"), "--variant", v, "--seed", str(seed), "--pool-root", str(t / "c"),
                           "--pool-geom", str(t / "g"), "--pool-tokens", str(t / "tok"), "--manifest", str(t / "m.json"),
                           "--pool-log-map", str(t / "map.json"), "--eval", "tune", "decision", "--epochs", "2", "--out", str(out / f"{v}_seed{seed}")]
                    if v == "c":
                        cmd.append("--shuffle-check")
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    self.assertEqual(res.returncode, 0, res.stderr[-1500:])
            z = np.load(out / "c_seed0" / "decision.npz")
            self.assertEqual(z["scores"].shape[1], 16)
            self.assertTrue((out / "c_seed0" / "tune_shuffled_tokens.npz").is_file())
            cmp = subprocess.run([sys.executable, str(HERE / "stageA_compare.py"), str(out), "decision", "--seeds", "0", "1"], capture_output=True, text=True)
            self.assertEqual(cmp.returncode, 0, cmp.stderr[-1500:])
            self.assertIn("c - b:", cmp.stdout); self.assertIn("c - d:", cmp.stdout)
            self.assertIn("97.5%", cmp.stdout)
            tune = set(np.load(out / "c_seed0" / "tune.npz")["tokens"].tolist()); dec = set(np.load(out / "c_seed0" / "decision.npz")["tokens"].tolist())
            self.assertFalse(tune & dec)                         # the two development subsets are disjoint
            self.assertFalse((tune | dec) & set(train))          # and disjoint from training


if __name__ == "__main__":
    unittest.main()
