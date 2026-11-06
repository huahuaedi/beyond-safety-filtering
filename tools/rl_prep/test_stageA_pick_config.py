import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import stageA_pick_config as P


class PickTest(unittest.TestCase):
    def test_best_per_variant_and_decision_never_read(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for name, score in (("c_lr0.0003_dr0", 0.80), ("c_lr0.001_dr0.1", 0.85), ("b_lr0.001_dr0", 0.70), ("b_lr0.0003_dr0.1", 0.65)):
                (t / name).mkdir()
                np.savez(t / name / "tune.npz", selected_pdms=np.full(5, score))
            (t / "c_lr0.001_dr0.1" / "decision.npz").write_bytes(b"not a valid npz")    # must not be opened
            table, chosen = P.pick(t)
            self.assertEqual((chosen["c"]["lr"], chosen["c"]["dropout"]), (0.001, 0.1))
            self.assertEqual((chosen["b"]["lr"], chosen["b"]["dropout"]), (0.001, 0.0))


if __name__ == "__main__":
    unittest.main()
