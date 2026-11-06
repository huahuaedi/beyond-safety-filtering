import json
import tempfile
import unittest
from pathlib import Path

import merge_policy_eval_shards as M

KEYS = M.COLUMNS


def shard(root, label, k, n, tokens, complete=True):
    d = Path(root) / f"{label}__shard{k}"
    (d / "validation").mkdir(parents=True)
    rows = [{"token": t, **{c: 0.5 for c in KEYS}} for t in tokens]
    (d / "validation" / "scores.json").write_text(json.dumps({"rows": rows}))
    (d / "summary.json").write_text(json.dumps({"complete": complete, "shard": [k, n]}))


class MergeTest(unittest.TestCase):
    def test_merge_and_checks(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            shard(t, "arm", 0, 2, ["a", "c"]); shard(t, "arm", 1, 2, ["b", "d"])
            n, m = M.merge(t, "arm", t / "out", expect=4)
            self.assertEqual(n, 4)
            self.assertAlmostEqual(m["score"], 0.5)
            self.assertEqual(len(json.loads((t / "out/validation/scores.json").read_text())["rows"]), 4)
            self.assertTrue((t / "out/scores.csv").read_text().strip().endswith(",0.5"))

    def test_incomplete_duplicate_and_missing_shard_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            shard(t, "a", 0, 2, ["x"], complete=False); shard(t, "a", 1, 2, ["y"])
            with self.assertRaises(ValueError):
                M.merge(t, "a", t / "o1")
            shard(t, "b", 0, 2, ["x"]); shard(t, "b", 1, 2, ["x"])
            with self.assertRaises(ValueError):
                M.merge(t, "b", t / "o2")
            shard(t, "c", 0, 3, ["x"]); shard(t, "c", 1, 3, ["y"])
            with self.assertRaises(ValueError):
                M.merge(t, "c", t / "o3")


if __name__ == "__main__":
    unittest.main()
