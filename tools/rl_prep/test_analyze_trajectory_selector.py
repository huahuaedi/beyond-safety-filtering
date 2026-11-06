import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_trajectory_selector as s


def make_scene(rng, n=16):
    """PDMS rises with path length, falls with deviation, so a selector can learn it."""
    t = np.arange(1, 9)[:, None]
    speed = rng.uniform(4, 9, size=n)
    lateral = rng.normal(0, 0.8, size=n)
    xy = np.stack([speed[:, None] * 0.5 * t.T, lateral[:, None] * (t.T / 8)], axis=-1)
    yaw = np.zeros((n, 8, 1))
    traj = np.concatenate([xy, yaw], axis=-1)
    ref = np.stack([np.concatenate([np.stack([6 * 0.5 * t[:, 0], 0 * t[:, 0]], -1), np.zeros((8, 1))], -1)] * 2)
    path = 0.5 * speed * 8
    score = np.clip(0.5 + 0.02 * path - 0.1 * np.abs(lateral), 0, 1)
    d = {"absolute_trajectories": traj, "reference_probe_absolute_trajectories": ref}
    for k in (*s.SAFETY, "comfort", "driving_direction_compliance"):
        d["score__" + k] = np.ones(n)
    d["score__ego_progress"] = score
    for k in (*s.SAFETY, "comfort"):
        d["reference_probe__" + k] = np.ones(2)
    d["reference_probe__ego_progress"] = np.array([0.6, 0.6])
    return d


def make_geom_scene(rng, n=16):
    """Same layout as make_scene, but a wide lateral offset leaves the drivable area (DAC=0)."""
    d = make_scene(rng, n)
    lateral = d["absolute_trajectories"][:, -1, 1]
    off = (np.abs(lateral) > 1.0).astype(float)
    d["score__drivable_area_compliance"] = 1.0 - off
    geom = np.zeros((n, 12))
    geom[:, 0] = rng.uniform(1.0, 5.0, n)           # min_clearance, unrelated
    geom[:, 8] = off * rng.uniform(0.2, 1.0, n)     # offroad_frac_max
    geom[:, 10] = off                               # offroad_frac_final
    return d, geom


class SelectorTest(unittest.TestCase):
    def write(self, root, split, name, rng):
        (root / split).mkdir(parents=True, exist_ok=True)
        np.savez(root / split / f"{name}.npz", **make_scene(rng))

    def test_end_to_end_learns_planted_signal(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "data"
            for i in range(60):
                self.write(root, "train", f"tr{i}", rng)
            for i in range(30):
                self.write(root, "test", f"te{i}", rng)
            man = Path(t) / "m.json"
            man.write_text(json.dumps({"train": [{"token": f"tr{i}"} for i in range(60)]}))
            out = Path(t) / "o.json"
            import sys
            old = sys.argv
            sys.argv = ["x", "--root", str(root), "--manifest", str(man), "--output", str(out)]
            try:
                s.main()
            finally:
                sys.argv = old
            rep = json.loads(out.read_text())
        h = rep["heldout"]
        self.assertEqual(h["scenes"], 30)
        self.assertGreater(h["selector"]["gain_vs_ref_x100"], h["random"]["gain_vs_ref_x100"] + 2)
        self.assertLessEqual(h["selector"]["gain_vs_ref_x100"], h["oracle"]["gain_vs_ref_x100"] + 1e-9)
        self.assertGreater(h["mean_within_scene_spearman"], 0.5)

    def test_no_signal_gives_no_gain(self):
        rng = np.random.default_rng(1)
        def noisy(r):
            d = make_scene(r)
            d["score__ego_progress"] = r.uniform(0.3, 1.0, size=16)  # unrelated to geometry
            return d
        feats = [noisy(rng) for _ in range(80)]
        paths = []
        with tempfile.TemporaryDirectory() as t:
            for i, d in enumerate(feats):
                p = Path(t) / f"{i}.npz"
                np.savez(p, **d)
                paths.append(p)
            x, y, g, meta = s.build(paths[:50])
            model = s.Ridge(1.0).fit(x, y)
            _, _, _, held = s.build(paths[50:])
            r = s.evaluate(model, held, np.random.default_rng(0))
        # selector should be near random, far from oracle
        self.assertLess(r["selector"]["gain_vs_ref_x100"] - r["random"]["gain_vs_ref_x100"],
                        0.5 * (r["oracle"]["gain_vs_ref_x100"] - r["random"]["gain_vs_ref_x100"]))

    def test_geometry_reduces_failures_that_trajectory_alone_cannot_see(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "data"; groot = Path(t) / "geom"
            for split, count in (("train", 80), ("test", 40)):
                (root / split).mkdir(parents=True); (groot / split).mkdir(parents=True)
                for i in range(count):
                    d, geom = make_geom_scene(rng)
                    # flatten the lateral signal in the PDMS target so only geometry knows who fails
                    np.savez(root / split / f"{split}{i}.npz", **d)
                    np.savez(groot / split / f"{split}{i}.npz", geom_features=geom,
                             feature_names=np.array(["x"] * 12))
            man = Path(t) / "m.json"
            man.write_text(json.dumps({"train": [{"token": f"train{i}"} for i in range(80)]}))
            out = Path(t) / "o.json"
            import sys
            old = sys.argv
            sys.argv = ["x", "--root", str(root), "--manifest", str(man), "--output", str(out),
                        "--geom-root", str(groot)]
            try:
                s.main()
            finally:
                sys.argv = old
            rep = json.loads(out.read_text())
        fs = rep["feature_sets"]
        self.assertEqual(set(fs), {"traj", "geom", "traj+geom"})
        both = fs["traj+geom"]["heldout"]
        # the picked candidate must fail DAC less often than a random one
        self.assertLess(both["failure_rate_of_pick"]["drivable_area_compliance"],
                        both["failure_rate_random"]["drivable_area_compliance"])
        self.assertGreater(both["selector"]["gain_vs_ref_x100"], both["random"]["gain_vs_ref_x100"])
        kc = both["k_curve"]
        self.assertEqual(set(kc), {"2", "4", "8", "16"})            # scene has 16 candidates
        self.assertGreaterEqual(kc["16"]["selector_gain_x100"], kc["2"]["selector_gain_x100"] - 1e-9)
        self.assertAlmostEqual(kc["16"]["selector_gain_x100"], both["selector"]["gain_vs_ref_x100"], places=6)
        self.assertEqual(set(both["pick_metric_delta_vs_ref_x100"]),
                         {"no_at_fault_collisions", "drivable_area_compliance",
                          "time_to_collision_within_bound", "ego_progress"})

    def test_hgb_option_runs_and_beats_random(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "data"; groot = Path(t) / "geom"
            for split, count in (("train", 60), ("test", 30)):
                (root / split).mkdir(parents=True); (groot / split).mkdir(parents=True)
                for i in range(count):
                    d, geom = make_geom_scene(rng)
                    np.savez(root / split / f"{split}{i}.npz", **d)
                    np.savez(groot / split / f"{split}{i}.npz", geom_features=geom,
                             feature_names=np.array(["x"] * 12))
            man = Path(t) / "m.json"
            man.write_text(json.dumps({"train": [{"token": f"train{i}"} for i in range(60)]}))
            out = Path(t) / "o.json"
            import sys
            old = sys.argv
            sys.argv = ["x", "--root", str(root), "--manifest", str(man), "--output", str(out),
                        "--geom-root", str(groot), "--hgb", "--hgb-iters", "10,30"]
            try:
                s.main()
            finally:
                sys.argv = old
            rep = json.loads(out.read_text())
        h = rep["feature_sets"]["traj+geom|hgb"]
        self.assertIn(h["chosen_iters"], (10, 30))
        self.assertGreater(h["heldout"]["selector"]["gain_vs_ref_x100"], h["heldout"]["random"]["gain_vs_ref_x100"])

    def test_extra_blocks_veto_and_shuffle_control(self):
        """A planted TTC failure that only the extra block can see; veto must remove it."""
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "data"; groot = Path(t) / "geom"; xroot = Path(t) / "ttc"; wroot = Path(t) / "wm"
            for split, count in (("train", 80), ("test", 40)):
                for r in (root, groot, xroot, wroot):
                    (r / split).mkdir(parents=True)
                for i in range(count):
                    d, geom = make_geom_scene(rng)
                    speed = d["absolute_trajectories"][:, -1, 0]
                    bad = (speed > np.median(speed)).astype(float)       # fast candidates hit a lead car
                    d["score__time_to_collision_within_bound"] = 1.0 - bad
                    d["score__ego_progress"] = d["score__ego_progress"] + 0.1 * bad  # tempting to pick
                    name = f"{split}{i}"
                    np.savez(root / split / f"{name}.npz", **d)
                    np.savez(groot / split / f"{name}.npz", geom_features=geom, feature_names=np.array(["x"] * 12))
                    ttc = np.zeros((16, 6)); ttc[:, 0] = bad
                    np.savez(xroot / split / f"{name}.npz", features=ttc)
                    np.savez(wroot / split / f"{name}.npz", features=rng.normal(size=(16, 4)))
            man = Path(t) / "m.json"
            man.write_text(json.dumps({"train": [{"token": f"train{i}"} for i in range(80)]}))
            cache = Path(t) / "cache"
            for i in range(40):
                d = cache / f"log{i % 5}" / "unknown" / f"test{i}"
                d.mkdir(parents=True)
                (d / "metric_cache.pkl").touch()
            out = Path(t) / "o.json"
            import sys
            old = sys.argv
            sys.argv = ["x", "--root", str(root), "--manifest", str(man), "--output", str(out),
                        "--geom-root", str(groot), "--extra-root", f"ttc={xroot}", "--extra-root", f"wm={wroot}",
                        "--shuffle-control", "wm", "--metric-cache-dirs", str(cache)]
            try:
                s.main()
            finally:
                sys.argv = old
            rep = json.loads(out.read_text())
        fs = rep["feature_sets"]
        for name in ("traj+geom+ttc", "traj+geom+wm", "traj+wm", "traj+geom+wm_shuf", "traj+geom+ttc+wm"):
            self.assertIn(name, fs)
        self.assertNotIn("traj+ttc", fs)
        h = fs["traj+geom+ttc"]["heldout"]
        ttc_fail = "time_to_collision_within_bound"
        self.assertEqual(h["veto"]["failure_rate_of_pick"][ttc_fail], 0.0)           # veto removes every planted failure
        self.assertGreater(fs["traj+geom"]["heldout"]["failure_rate_of_pick"][ttc_fail], 0.0)
        self.assertGreater(h["veto"]["mean_fraction_of_candidates_vetoed"], 0.3)
        # paired comparison, log-cluster intervals and the selector's own fix/break table
        pr = rep["paired"]["traj+geom+ttc minus traj+geom"]
        self.assertEqual(pr["scenes"], 40)
        self.assertEqual(pr["logs"], 5)
        lo, hi = pr["log_cluster_ci95_x100"]
        self.assertLessEqual(lo, pr["mean_diff_x100"] + 1e-9)
        self.assertGreaterEqual(hi, pr["mean_diff_x100"] - 1e-9)
        self.assertIn("traj+geom+wm minus traj+geom+wm_shuf", rep["paired"])
        self.assertIn("log_cluster_ci95_x100", fs["traj+geom"]["heldout"]["selector"])
        fb = fs["traj+geom"]["heldout"]["fix_break"][ttc_fail]
        self.assertEqual(set(fb), {"ref_failures", "fixed", "broken", "p_fix", "p_break", "net_passing_scenes"})
        self.assertEqual(fb["net_passing_scenes"], fb["fixed"] - fb["broken"])
        # the TTC block lets the selector avoid broken scenes it otherwise creates
        self.assertLessEqual(h["fix_break"][ttc_fail]["broken"], fs["traj+geom"]["heldout"]["fix_break"][ttc_fail]["broken"])

    def test_structured_scorer_and_set_features(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "data"; groot = Path(t) / "geom"
            for split, count in (("train", 70), ("test", 30)):
                (root / split).mkdir(parents=True); (groot / split).mkdir(parents=True)
                for i in range(count):
                    d, geom = make_geom_scene(rng)
                    np.savez(root / split / f"{split}{i}.npz", **d)
                    np.savez(groot / split / f"{split}{i}.npz", geom_features=geom, feature_names=np.array(["x"] * 12))
            man = Path(t) / "m.json"
            man.write_text(json.dumps({"train": [{"token": f"train{i}"} for i in range(70)]}))
            out = Path(t) / "o.json"
            import sys
            old = sys.argv
            sys.argv = ["x", "--root", str(root), "--manifest", str(man), "--output", str(out),
                        "--geom-root", str(groot), "--hgb", "--struct", "--set-features", "--hgb-iters", "10,30"]
            try:
                s.main()
            finally:
                sys.argv = old
            rep = json.loads(out.read_text())
        fs = rep["feature_sets"]
        self.assertIn("traj+geom|struct", fs)
        h = fs["traj+geom|struct"]["heldout"]
        self.assertIn(fs["traj+geom|struct"]["chosen_iters"], (10, 30))
        self.assertGreater(h["selector"]["gain_vs_ref_x100"], h["random"]["gain_vs_ref_x100"])
        self.assertIn("traj+geom|struct minus traj+geom|hgb", rep["paired"])

    def test_set_level_features_shape_and_ranks(self):
        rng = np.random.default_rng(3)
        d = make_scene(rng, n=16)
        base = s.scene_features(d["absolute_trajectories"], d["reference_probe_absolute_trajectories"])
        withset = s.scene_features(d["absolute_trajectories"], d["reference_probe_absolute_trajectories"], set_features=True)
        self.assertEqual(withset.shape[1], base.shape[1] + 4)
        self.assertTrue(np.allclose(withset[:, : base.shape[1]], base))
        ranks = withset[:, -2]                       # path-length rank in [0, 1]
        self.assertAlmostEqual(float(ranks.min()), 0.0)
        self.assertAlmostEqual(float(ranks.max()), 1.0)

    def test_train_heldout_never_overlap(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for split, names in (("train", ["a", "b"]), ("val", ["c"]), ("test", ["d"])):
                (root / split).mkdir()
                for n in names:
                    (root / split / f"{n}.npz").touch()
            tr, he = s.split_paths(root, {"train": [{"token": "a"}, {"token": "c"}]})
        self.assertEqual({p.stem for p in tr}, {"a", "c"})
        self.assertEqual({p.stem for p in he}, {"b", "d"})
        self.assertFalse({p.stem for p in tr} & {p.stem for p in he})


if __name__ == "__main__":
    unittest.main()
