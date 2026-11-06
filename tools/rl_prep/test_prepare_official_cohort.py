import json
import pickle
import tempfile
import unittest
from pathlib import Path

import prepare_official_cohort as P


class PrepareTest(unittest.TestCase):
    def test_tokens_length_and_log_map(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "f.yaml").write_text("tokens:\n  - aa\n  - bb\n  - cc\n")
            self.assertEqual(P.navtest_tokens(t / "f.yaml"), ["aa", "bb", "cc"])
            pickle.dump({"data": [1, 2, 3]}, open(t / "x.pkl", "wb"))
            self.assertEqual(P.pickle_length(t / "x.pkl"), 3)
            pickle.dump([1, 2], open(t / "y.pkl", "wb"))
            self.assertEqual(P.pickle_length(t / "y.pkl"), 2)
            for log, tok in (("log1", "aa"), ("log1", "bb"), ("log2", "cc"), ("log3", "zz")):
                d = t / "cache" / log / "type" / tok
                d.mkdir(parents=True)
                (d / "metric_cache.pkl").write_bytes(b"x")
            m = P.token_log_map(t / "cache", ["aa", "bb", "cc"])
            self.assertEqual(m, {"aa": "log_name:log1", "bb": "log_name:log1", "cc": "log_name:log2"})

    def test_add_tokens(self):
        recs = [{"a": 1}, {"a": 2}]
        out = P.add_tokens(recs, ["x", "y"])
        self.assertEqual([r["token"] for r in out], ["x", "y"])
        self.assertNotIn("token", recs[0])  # input untouched
        self.assertEqual(P.add_tokens({"data": recs, "k": 1}, ["x", "y"])["k"], 1)
        with self.assertRaises(ValueError):
            P.add_tokens(recs, ["x"])
        with self.assertRaises(ValueError):
            P.add_tokens([{"token": "q"}], ["x"])

    def test_duplicate_tokens_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            Path(t, "f.yaml").write_text("tokens:\n  - aa\n  - aa\n")
            with self.assertRaises(ValueError):
                P.navtest_tokens(Path(t) / "f.yaml")


if __name__ == "__main__":
    unittest.main()
