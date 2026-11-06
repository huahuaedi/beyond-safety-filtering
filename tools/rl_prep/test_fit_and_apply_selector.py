import csv, json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
from test_analyze_trajectory_selector import make_geom_scene

HERE = Path(__file__).parent


class FitApplyTest(unittest.TestCase):
    def test_fit_on_pool_apply_to_new_cohort(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for tag, n in (("pool", 40), ("cohort", 24)):
                for split in ("train", "val", "test"):
                    (t / tag / split).mkdir(parents=True); (t / (tag + "_geom") / split).mkdir(parents=True)
                for i in range(n):
                    d, geom = make_geom_scene(rng)
                    split = ("train", "val", "test")[i % 3]
                    np.savez(t / tag / split / f"{tag}{i}.npz", **d)
                    np.savez(t / (tag + "_geom") / split / f"{tag}{i}.npz", geom_features=geom,
                             feature_names=np.array(["x"] * 12))
            (t / "map.json").write_text(json.dumps({f"cohort{i}": f"log_name:L{i % 8}" for i in range(24)}))
            with open(t / "sft.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["", "token", "valid", "score"])
                for i in range(24):
                    w.writerow([i, f"cohort{i}", "True", 0.6])
                w.writerow(["", "average", "", 0.6])
            out = t / "o.json"
            res = subprocess.run([sys.executable, str(HERE / "fit_and_apply_selector.py"),
                                  "--train-root", str(t / "pool"), "--train-geom-root", str(t / "pool_geom"),
                                  "--apply-root", str(t / "cohort"), "--apply-geom-root", str(t / "cohort_geom"),
                                  "--token-log-map", str(t / "map.json"), "--sft-csv", str(t / "sft.csv"),
                                  "--hgb-iters", "20", "--model-out", str(t / "m.pkl"), "--output", str(out)],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            r = json.loads(out.read_text())
            self.assertTrue((t / "m.pkl").is_file())
        h = r["models"]["hgb"]
        self.assertEqual(r["apply_scenes"], 24)
        self.assertEqual(r["apply_logs"], 8)
        self.assertGreater(h["gain_vs_reference_probes_x100"], h["random_gain_x100"])
        self.assertLessEqual(h["gain_vs_reference_probes_x100"], h["oracle_gain_x100"] + 1e-9)
        self.assertEqual(h["independent_sft_eval_scenes"], 24)
        d = h["detail"]
        self.assertEqual(set(d["component_vs_reference_probes"]), {"no_at_fault_collisions", "drivable_area_compliance",
                                                                   "time_to_collision_within_bound", "ego_progress"})
        self.assertEqual(d["vs_independent_sft"]["scenes"], 24)
        self.assertEqual(len(h["per_scene"]), 24)
        fb = d["fix_break_vs_reference_probes"]["drivable_area_compliance"]
        self.assertEqual(fb["net_passing_scenes"], fb["fixed"] - fb["broken"])
        self.assertEqual(set(r["baselines"]), {"random_candidate", "closest_to_reference", "longest_path",
                                               "geometric_veto_then_closest", "oracle"})
        self.assertEqual(set(r["ablations"]), {"ridge_traj_only", "hgb_traj_only", "ridge_geom_only", "hgb_geom_only"})
        self.assertGreaterEqual(r["baselines"]["oracle"]["gain_vs_reference_probes_x100"],
                                h["gain_vs_reference_probes_x100"] - 1e-9)
        self.assertEqual(r["candidates_per_scene"], 16)
        lo, hi = h["gain_log_ci95_x100"]
        self.assertLessEqual(lo, h["gain_vs_reference_probes_x100"] + 1e-9)
        self.assertGreaterEqual(hi, h["gain_vs_reference_probes_x100"] - 1e-9)

    def test_split_geometry_adds_map_and_agent_ablations(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for tag, n in (("pool", 40), ("cohort", 20)):
                for split in ("train", "val", "test"):
                    (t / tag / split).mkdir(parents=True); (t / (tag + "_geom") / split).mkdir(parents=True)
                for i in range(n):
                    d, geom = make_geom_scene(rng)
                    split = ("train", "val", "test")[i % 3]
                    np.savez(t / tag / split / f"{tag}{i}.npz", **d)
                    np.savez(t / (tag + "_geom") / split / f"{tag}{i}.npz", geom_features=geom,
                             feature_names=np.array(["x"] * 12))
            (t / "map.json").write_text(json.dumps({f"cohort{i}": f"log_name:L{i % 5}" for i in range(20)}))
            out = t / "o.json"
            res = subprocess.run([sys.executable, str(HERE / "fit_and_apply_selector.py"),
                                  "--train-root", str(t / "pool"), "--train-geom-root", str(t / "pool_geom"),
                                  "--apply-root", str(t / "cohort"), "--apply-geom-root", str(t / "cohort_geom"),
                                  "--token-log-map", str(t / "map.json"), "--hgb-iters", "10", "--split-geometry",
                                  "--output", str(out)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            r = json.loads(out.read_text())
        names = set(r["ablations"])
        for tag in ("map", "agent", "traj_map", "traj_agent", "traj", "geom"):
            self.assertIn(f"hgb_{tag}_only", names)
            self.assertIn(f"ridge_{tag}_only", names)

    def test_extra_feature_blocks_are_used_and_must_match(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for tag, n in (("pool", 40), ("cohort", 20)):
                for split in ("train", "val", "test"):
                    for sub in ("", "_geom", "_x"):
                        (t / (tag + sub) / split).mkdir(parents=True)
                for i in range(n):
                    d, geom = make_geom_scene(rng)
                    split = ("train", "val", "test")[i % 3]
                    np.savez(t / tag / split / f"{tag}{i}.npz", **d)
                    np.savez(t / (tag + "_geom") / split / f"{tag}{i}.npz", geom_features=geom, feature_names=np.array(["x"] * 12))
                    np.savez(t / (tag + "_x") / split / f"{tag}{i}.npz", features=rng.normal(size=(16, 2)))
            (t / "map.json").write_text(json.dumps({f"cohort{i}": f"log_name:L{i % 5}" for i in range(20)}))
            base = [sys.executable, str(HERE / "fit_and_apply_selector.py"), "--train-root", str(t / "pool"),
                    "--train-geom-root", str(t / "pool_geom"), "--apply-root", str(t / "cohort"), "--apply-geom-root", str(t / "cohort_geom"),
                    "--token-log-map", str(t / "map.json"), "--hgb-iters", "10"]
            ok = subprocess.run(base + ["--train-extra-root", f"anchor={t / 'pool_x'}", "--apply-extra-root", f"anchor={t / 'cohort_x'}",
                                        "--output", str(t / "o.json")], capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertEqual(json.loads((t / "o.json").read_text())["apply_scenes"], 20)
            bad = subprocess.run(base + ["--train-extra-root", f"anchor={t / 'pool_x'}", "--output", str(t / "o2.json")], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)

    def test_split_extras_adds_block_ablations(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for tag, n in (("pool", 40), ("cohort", 20)):
                for split in ("train", "val", "test"):
                    for sub in ("", "_geom", "_x"):
                        (t / (tag + sub) / split).mkdir(parents=True)
                for i in range(n):
                    d, geom = make_geom_scene(rng)
                    split = ("train", "val", "test")[i % 3]
                    np.savez(t / tag / split / f"{tag}{i}.npz", **d)
                    np.savez(t / (tag + "_geom") / split / f"{tag}{i}.npz", geom_features=geom, feature_names=np.array(["x"] * 12))
                    np.savez(t / (tag + "_x") / split / f"{tag}{i}.npz", features=np.repeat(rng.normal(size=(1, 3)), 16, axis=0))
            (t / "map.json").write_text(json.dumps({f"cohort{i}": f"log_name:L{i % 5}" for i in range(20)}))
            res = subprocess.run([sys.executable, str(HERE / "fit_and_apply_selector.py"), "--train-root", str(t / "pool"),
                                  "--train-geom-root", str(t / "pool_geom"), "--apply-root", str(t / "cohort"),
                                  "--apply-geom-root", str(t / "cohort_geom"), "--token-log-map", str(t / "map.json"), "--hgb-iters", "10",
                                  "--train-extra-root", f"pv={t / 'pool_x'}", "--apply-extra-root", f"pv={t / 'cohort_x'}",
                                  "--split-geometry", "--split-extras", "--output", str(t / "o.json")], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            names = set(json.loads((t / "o.json").read_text())["ablations"])
        for tag in ("hgb_pv_only", "hgb_traj_pv_only", "hgb_traj_map_pv_only", "ridge_traj_map_pv_only"):
            self.assertIn(tag, names)

    def test_missing_log_mapping_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            rng = np.random.default_rng(0)
            for tag in ("pool", "cohort"):
                for split in ("train", "val", "test"):
                    (t / tag / split).mkdir(parents=True); (t / (tag + "_geom") / split).mkdir(parents=True)
                d, geom = make_geom_scene(rng)
                np.savez(t / tag / "train" / f"{tag}0.npz", **d)
                np.savez(t / (tag + "_geom") / "train" / f"{tag}0.npz", geom_features=geom, feature_names=np.array(["x"] * 12))
            (t / "map.json").write_text("{}")
            res = subprocess.run([sys.executable, str(HERE / "fit_and_apply_selector.py"),
                                  "--train-root", str(t / "pool"), "--train-geom-root", str(t / "pool_geom"),
                                  "--apply-root", str(t / "cohort"), "--apply-geom-root", str(t / "cohort_geom"),
                                  "--token-log-map", str(t / "map.json"), "--output", str(t / "o.json")],
                                 capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("log mapping", res.stderr)


if __name__ == "__main__":
    unittest.main()
