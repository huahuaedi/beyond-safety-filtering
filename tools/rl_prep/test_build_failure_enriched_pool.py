import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import build_failure_enriched_pool as b

K = b.SAFETY


def write(path, fail_count, n=64, ref_dac=1.0):
    d = {}
    for k in K:
        d["score__" + k] = np.ones(n, np.float32)
        d["reference_probe__" + k] = np.ones(2, np.float32)
    d["score__drivable_area_compliance"][:fail_count] = 0.0
    d["reference_probe__drivable_area_compliance"] = np.full(2, ref_dac, np.float32)
    np.savez(path, **d)


class PoolTest(unittest.TestCase):
    def test_rate_uses_reference_relative_rule(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "a.npz"
            write(p, 8)                                   # 8/64 candidates fail DAC, reference is clean
            rate, unsafe = b.scene_rate(p)
            self.assertAlmostEqual(rate, 8 / 64)
            self.assertFalse(unsafe)
            write(p, 8, ref_dac=0.0)                      # reference already fails DAC: candidates are not regressions
            rate, unsafe = b.scene_rate(p)
            self.assertEqual(rate, 0.0)
            self.assertTrue(unsafe)

    def test_pools_and_expected_uninformative_share(self):
        rates = {"clean": 0.0, "rare": 1 / 64, "hot": 16 / 64}
        unsafe = {"clean": False, "rare": False, "hot": False}
        tokens, chosen, hit = b.pools(rates, unsafe, 16, 0.75)
        self.assertEqual(chosen, ["hot"])
        full = b.summarize("all", tokens, hit, unsafe)["expected_share_of_all_feasible_groups"]
        enr = b.summarize("e", chosen, hit, unsafe)["expected_share_of_all_feasible_groups"]
        self.assertLess(enr, full)
        self.assertAlmostEqual(hit["hot"], 1 - 0.75 ** 16)

    def test_ref_unsafe_scene_is_always_kept_and_main_runs(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "train").mkdir()
            write(t / "train" / "clean.npz", 0)
            write(t / "train" / "unsafe.npz", 0, ref_dac=0.0)
            write(t / "train" / "hot.npz", 20)
            man = t / "m.json"
            man.write_text(json.dumps({"schema_version": 2, "train": [{"token": x, "split": "train"} for x in ("clean", "unsafe", "hot")]}))
            import sys
            old = sys.argv
            sys.argv = ["x", "--root", str(t), "--manifest", str(man), "--output-dir", str(t / "out")]
            try:
                b.main()
            finally:
                sys.argv = old
            chosen = json.loads((t / "out" / "pool_failure_enriched.json").read_text())
        self.assertEqual(chosen, ["hot", "unsafe"])

    def test_rejects_non_log_disjoint_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            man = t / "m.json"
            man.write_text(json.dumps({"schema_version": 1, "train": []}))
            import sys
            old = sys.argv
            sys.argv = ["x", "--root", str(t), "--manifest", str(man), "--output-dir", str(t / "o")]
            try:
                with self.assertRaises(ValueError):
                    b.main()
            finally:
                sys.argv = old


if __name__ == "__main__":
    unittest.main()
