import tempfile
import unittest
from pathlib import Path

import numpy as np

import make_policy_visual_blocks as M


class BlocksTest(unittest.TestCase):
    def test_pca_unit_variance_and_projection_shape(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 12)) * np.arange(1, 13)
        pca = M.fit_pca(x, 4)
        z = M.project(pca, x)
        self.assertEqual(z.shape, (200, 4))
        np.testing.assert_allclose(z.std(0, ddof=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-6)

    def test_load_contexts_rejects_duplicates_and_blocks_repeat_per_candidate(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            rng = np.random.default_rng(1)
            np.savez(t / "context_shard0.npz", tokens=np.array(["a", "b"]), contexts=rng.normal(size=(2, 6)).astype(np.float16))
            np.savez(t / "context_shard1.npz", tokens=np.array(["c", "d"]), contexts=rng.normal(size=(2, 6)).astype(np.float16))
            tokens, x = M.load_contexts(t)
            self.assertEqual(tokens, ["a", "b", "c", "d"])
            self.assertEqual(x.shape, (4, 6))
            np.savez(t / "context_shard2.npz", tokens=np.array(["a"]), contexts=np.zeros((1, 6), np.float16))
            with self.assertRaises(ValueError):
                M.load_contexts(t)


if __name__ == "__main__":
    unittest.main()
