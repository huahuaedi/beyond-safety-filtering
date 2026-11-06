"""Flow test for the metric-cache adapter using stub nuPlan/NAVSIM objects.

The stubs mirror the attribute names the adapter *assumes* of the real
classes, so this checks the adapter's logic (t=0 filtering, sharding, resume,
failure on missing caches), not the real NAVSIM API.  A real-cache smoke test
is the ``--limit`` mode of ``submit_extract_geometric_features.slurm``.
"""
import lzma
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

import extract_geometric_features as ext
import geometric_features as gf


class FakeMap:
    def __init__(self, tokens):
        self.tokens = tokens


class FakeObservation:
    def __init__(self, present, unique):
        self._present, self.unique_objects = present, unique

    def __getitem__(self, i):
        assert i == 0
        return FakeMap(self._present)


class FakeDrivable:
    def points_in_polygons(self, points):
        return (np.abs(points[..., 1]) < 4.0)[None]  # one polygon: |y| < 4


def obj(x, y, vx=0.0, vy=0.0, heading=0.0, moving=True):
    box = NS(center=NS(x=x, y=y, heading=heading), length=4.0, width=2.0)
    return NS(box=box, velocity=NS(x=vx, y=vy)) if moving else NS(box=box)


def make_cache(unique, present):
    params = NS(length=4.6, width=2.0, front_length=3.5, rear_length=1.1)
    ego = NS(
        car_footprint=NS(vehicle_parameters=params),
        center=NS(x=1.4, y=0.0, heading=0.0), rear_axle=NS(x=0.0, y=0.0, heading=0.0),
    )
    return NS(
        observation=FakeObservation(present, unique), drivable_area_map=FakeDrivable(), ego_state=ego
    )


class FakeLoader:
    root = None

    def __init__(self, path):
        self.metric_cache_paths = {p.stem: p for p in Path(path).glob("*.pkl")}


class ExtractorFlowTest(unittest.TestCase):
    def setUp(self):
        stub = types.ModuleType("navsim.common.dataloader")
        stub.MetricCacheLoader = FakeLoader
        self._saved = {k: sys.modules.get(k) for k in ("navsim", "navsim.common", "navsim.common.dataloader")}
        sys.modules["navsim"] = types.ModuleType("navsim")
        sys.modules["navsim.common"] = types.ModuleType("navsim.common")
        sys.modules["navsim.common.dataloader"] = stub

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    def build(self, root, tokens, cache_tokens=None):
        rng = np.random.default_rng(0)
        (root / "data" / "train").mkdir(parents=True)
        (root / "cache").mkdir()
        for token in tokens:
            cands = np.stack(
                [np.stack([s * gf.HORIZON_TIMES, np.zeros(8), np.zeros(8)], -1) for s in rng.uniform(0, 6, 64)]
            ).astype("float32")
            np.savez(root / "data" / "train" / f"{token}.npz", scene_token=token, absolute_trajectories=cands)
        for token in cache_tokens if cache_tokens is not None else tokens:
            cache = make_cache(
                {"a": obj(15.0, 0.0), "future_only": obj(3.0, 0.0), "ped": obj(20.0, 1.0, moving=False)},
                present=["a", "ped"],
            )
            with lzma.open(root / "cache" / f"{token}.pkl", "wb") as handle:
                handle.write(pickle.dumps(cache))

    def run_main(self, root, *extra):
        ext.main([
            "--data-root", str(root / "data"), "--metric-cache-path", str(root / "cache"),
            "--output-root", str(root / "out"), *extra,
        ])

    def test_t0_filter_output_shape_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, ["s0", "s1", "s2"])
            self.run_main(root)
            with np.load(root / "out" / "train" / "s0.npz") as out:
                self.assertEqual(out["geom_features"].shape, (64, len(gf.FEATURE_NAMES)))
                self.assertEqual(int(out["num_agents"]), 2)  # 'future_only' must be dropped
                self.assertEqual(str(out["scene_token"]), "s0")
                self.assertTrue(np.isfinite(out["geom_features"]).all())
                first = out["geom_features"].copy()
            # Faster candidates approach the obstacle at 15 m more closely.
            self.assertLess(first[-1, 0] + 1e-6, 10.0)
            stamp = (root / "out" / "train" / "s1.npz").stat().st_mtime_ns
            self.run_main(root)  # resume: existing outputs are left untouched
            self.assertEqual((root / "out" / "train" / "s1.npz").stat().st_mtime_ns, stamp)

    def test_shards_partition_the_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, [f"s{i}" for i in range(5)])
            self.run_main(root, "--shard-id", "0", "--num-shards", "2")
            self.run_main(root, "--shard-id", "1", "--num-shards", "2")
            self.assertEqual(len(list((root / "out" / "train").glob("*.npz"))), 5)

    def test_missing_metric_cache_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, ["s0", "s1"], cache_tokens=["s0"])
            with self.assertRaises(SystemExit) as raised:
                self.run_main(root)
            self.assertIn("no metric cache", str(raised.exception))

    def test_token_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, ["s0"])
            (root / "data" / "train" / "s0.npz").rename(root / "data" / "train" / "wrong.npz")
            with self.assertRaises(ValueError):
                self.run_main(root)


if __name__ == "__main__":
    unittest.main()
