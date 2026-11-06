import tempfile
import unittest
from pathlib import Path

import numpy as np

import stageA_data as D
from test_analyze_trajectory_selector import make_geom_scene


class DataTest(unittest.TestCase):
    def test_scene_data_shapes_and_labels(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "c" / "train").mkdir(parents=True); (t / "g" / "train").mkdir(parents=True)
            d, geom = make_geom_scene(rng)
            np.savez(t / "c" / "train" / "s0.npz", **d)
            np.savez(t / "g" / "train" / "s0.npz", geom_features=geom, feature_names=np.array(["x"] * 12))
            sd = D.load_scene_data(t / "c" / "train" / "s0.npz", t / "g")
        self.assertEqual(sd["traj"].shape, (16, 40)); self.assertEqual(sd["geom"].shape, (16, 15))
        self.assertEqual(sd["gates"].shape, (16, 3)); self.assertEqual(sd["cont"].shape, (16, 5))
        off = sd["geom"][:, 8] > 0
        np.testing.assert_array_equal(sd["gates"][:, 1] == 0, off)           # DAC fails exactly where the planted off-road is

    def test_split_logs_and_derangement(self):
        logs = [f"L{i % 10}" for i in range(100)]
        a, b = D.split_logs(logs, 0.7, 0)
        self.assertEqual(len(a), 7); self.assertFalse(a & b); self.assertEqual(a | b, set(logs))
        self.assertEqual(D.split_logs(logs, 0.7, 0)[0], a)
        j = D.cross_log_derangement(logs, 3)
        self.assertEqual(sorted(j.tolist()), list(range(100)))               # a permutation
        arr = np.array(logs)
        self.assertTrue((arr[j] != arr).all())                               # never the same log
        with self.assertRaises(ValueError):
            D.cross_log_derangement(["a"] * 5, 0)

    def test_token_shards_roundtrip_and_consistency(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            grid = np.stack([np.zeros(6, int), np.arange(6) // 3, np.arange(6) % 3], axis=1).astype(np.int16)
            for k, names in enumerate((["a", "b"], ["c"])):
                np.savez(t / f"tokens_shard{k}.npz", tokens=np.array(names), grid=grid,
                         h=np.random.default_rng(k).normal(size=(len(names), 6, 4)).astype(np.float16),
                         projection_fingerprint=-0.3)
            idx, h, g, fp = D.load_token_shards(t)
            self.assertEqual(idx, {"a": 0, "b": 1, "c": 2}); self.assertEqual(h.shape, (3, 6, 4)); self.assertEqual(fp, -0.3)
            np.savez(t / "tokens_shard2.npz", tokens=np.array(["d"]), grid=grid, h=np.zeros((1, 6, 4), np.float16), projection_fingerprint=0.5)
            with self.assertRaises(ValueError):
                D.load_token_shards(t)


if __name__ == "__main__":
    unittest.main()
