import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import export_selected_to_official_json as E
from offline_verified_data import absolute_from_relative


def write_scene(root, token, n=3, corrupt=False):
    rng = np.random.default_rng(abs(hash(token)) % 1000)
    rel = (rng.normal(size=(n, 8, 3)) * 0.3).astype(np.float32)
    ab = absolute_from_relative(rel)
    if corrupt:
        ab = ab + 1.0
    d = Path(root) / "train"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{token}.npz", relative_trajectories=rel, absolute_trajectories=ab)
    return rel


class ExportTest(unittest.TestCase):
    def test_agent_conversion_matches_stored_absolute_and_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            rel = write_scene(t / "c", "a")
            write_scene(t / "c", "b")
            n = E.export([{"token": "a", "pick": 2}, {"token": "b", "pick": 0}], t / "c", t / "out")
            self.assertEqual(n, 2)
            got = np.array(json.loads((t / "out" / "a.json").read_text())["action"])
            np.testing.assert_allclose(got, rel[2], atol=1e-6)

    def test_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write_scene(t / "c", "a", corrupt=True)
            with self.assertRaises(ValueError):
                E.export([{"token": "a", "pick": 0}], t / "c", t / "out")

    def test_refuses_existing_output_dir(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write_scene(t / "c", "a")
            (t / "out").mkdir()
            with self.assertRaises(FileExistsError):
                E.export([{"token": "a", "pick": 0}], t / "c", t / "out")


if __name__ == "__main__":
    unittest.main()
