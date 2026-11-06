import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import extract_wm_pca_features as E


class WmPcaTest(unittest.TestCase):
    def test_features_follow_the_within_scene_signal_and_fit_uses_only_train(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            lat, out = t / "lat", t / "out"
            direction = rng.normal(size=64)
            for split, n in (("train", 20), ("test", 8)):
                (lat / split).mkdir(parents=True)
                for i in range(n):
                    strength = rng.normal(size=(16, 1))
                    ctx = rng.normal(size=(1, 1, 64)) * 5.0                     # shared scene context
                    z = ctx + strength[:, None, :] * direction[None, None, :] + 0.05 * rng.normal(size=(16, 2, 64))
                    np.savez(lat / split / f"{split}{i}.npz", eoa_latents=z.astype(np.float16),
                             _strength=strength)
            man = t / "m.json"
            man.write_text(json.dumps({"train": [{"token": f"train{i}"} for i in range(20)]}))
            E.main(["--latent-root", str(lat), "--manifest", str(man), "--out-root", str(out),
                    "--layers", "0,1", "--k", "4", "--sample-per-scene", "8"])
            f = np.load(out / "L0" / "test" / "test0.npz")["features"]
            s = np.load(lat / "test" / "test0.npz")["_strength"][:, 0]
            self.assertEqual(f.shape, (16, 4))
            self.assertTrue((out / "L1" / "train" / "train3.npz").exists())
            # the first axis recovers the planted within-scene direction despite the large shared context
            self.assertGreater(abs(np.corrcoef(f[:, 0], s)[0, 1]), 0.9)

    def test_rejects_wrong_rank(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "a.npz"
            np.savez(p, eoa_latents=np.zeros((4, 8)))
            with self.assertRaises(ValueError):
                E.load_centred(p, [0])


if __name__ == "__main__":
    unittest.main()
