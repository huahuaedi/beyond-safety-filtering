import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import geometric_features as gf
import summarize_risk_baselines as summ

HAS_TORCH = importlib.util.find_spec("torch") is not None
REPO = Path(__file__).resolve().parents[2]
NF = len(gf.FEATURE_NAMES)


class SummaryMathTest(unittest.TestCase):
    def units(self, S=60, seed=0):
        rng = np.random.default_rng(seed)
        labels = (rng.random((S, 64, 4)) < 0.15).astype(np.float32)
        names = [f"s{i:03d}" for i in range(S)]
        perfect = labels + 0.01 * rng.random(labels.shape)
        noise = rng.random(labels.shape)
        return names, labels, perfect.astype(np.float32), noise.astype(np.float32)

    def test_perfect_ranker_beats_random_with_a_significant_paired_interval(self):
        names, labels, perfect, noise = self.units()
        report = summ.compare({"noise": [(names, noise, labels)], "perfect": [(names, perfect, labels)]},
                              reference_mode="noise", resamples=500, seed=1)
        cell = report["modes"]["perfect"]["recall_at_8_minus_reference"]["unsafe"]
        self.assertTrue(cell["excludes_zero"])
        self.assertGreater(cell["delta"], 0.3)
        self.assertGreater(report["modes"]["perfect"]["auroc"]["unsafe"], 0.99)
        # A random ranking sits near lift 1.
        self.assertLess(abs(report["modes"]["noise"]["lift_at_8"]["unsafe"] - 1.0), 0.25)

    def test_identical_predictions_have_zero_delta_and_include_zero(self):
        names, labels, _, noise = self.units()
        report = summ.compare({"a": [(names, noise, labels)], "b": [(names, noise.copy(), labels)]},
                              reference_mode="a", resamples=200, seed=1)
        cell = report["modes"]["b"]["recall_at_16_minus_reference"]["ttc_failure"]
        self.assertEqual(cell["delta"], 0.0)
        self.assertFalse(cell["excludes_zero"])

    def test_scene_order_is_aligned_and_mismatched_sets_are_rejected(self):
        names, labels, perfect, _ = self.units(S=20)
        order = np.random.default_rng(3).permutation(20)
        shuffled = ([names[i] for i in order], perfect[order], labels[order])
        report = summ.compare({"x": [(names, perfect, labels)], "y": [shuffled]}, "x", resamples=50)
        self.assertEqual(report["modes"]["y"]["recall_at_8_minus_reference"]["unsafe"]["delta"], 0.0)
        with self.assertRaises(ValueError):
            summ.compare({"x": [(names, perfect, labels)], "y": [(names[:-1] + ["other"], perfect, labels)]}, "x")

    def test_recall_and_capture_match_hand_computation(self):
        labels = np.zeros((1, 6, 1), dtype=np.float32)
        labels[0, [1, 4], 0] = 1
        prob = np.array([0.1, 0.9, 0.2, 0.3, 0.8, 0.0], dtype=np.float32).reshape(1, 6, 1)
        self.assertEqual(int(summ.captured_at_k(labels, prob, 2)[0, 0]), 2)
        self.assertEqual(int(summ.captured_at_k(labels, prob, 1)[0, 0]), 1)
        self.assertAlmostEqual(summ.auroc(labels[0, :, 0], prob[0, :, 0]), 1.0)
        self.assertAlmostEqual(summ.average_precision(labels[0, :, 0], prob[0, :, 0]), 1.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch required")
class BlockSemanticsTest(unittest.TestCase):
    def test_scene_and_residual_blocks(self):
        import torch
        import train_risk_baselines as trb

        rng = np.random.default_rng(0)
        latent = torch.from_numpy(rng.normal(size=(3, 64, trb.LATENT_DIM)).astype("float32"))
        batch = {"latent": latent}
        scene = trb.BlockRanker.block_input("scene1", batch)
        resid = trb.BlockRanker.block_input("resid1", batch)
        self.assertTrue(torch.allclose(scene[:, 0], scene[:, 17]))  # constant across candidates
        self.assertLess(float(resid.sum(dim=1).abs().max()), 1e-3)  # zero-mean within scene
        self.assertTrue(torch.allclose(scene + resid, latent, atol=1e-5))

    def test_residual_mode_ignores_scene_constant_shift_but_raw_latent_does_not(self):
        import torch
        import train_risk_baselines as trb

        torch.manual_seed(0)
        latent = torch.randn(2, 64, trb.LATENT_DIM)
        shift = torch.randn(2, 1, trb.LATENT_DIM) * 5
        resid_model = trb.BlockRanker(("resid1",)).eval()
        raw_model = trb.BlockRanker(("layer1",)).eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(resid_model({"latent": latent}), resid_model({"latent": latent + shift}), atol=1e-4))
            self.assertFalse(torch.allclose(raw_model({"latent": latent}), raw_model({"latent": latent + shift}), atol=1e-3))

    def test_unknown_block_rejected(self):
        import train_risk_baselines as trb

        with self.assertRaises(ValueError):
            trb.BlockRanker(("action", "oracle"))


def write_scene(root, split, name, rng, informative=True):
    (root / "data" / split).mkdir(parents=True, exist_ok=True)
    (root / "lat" / split).mkdir(parents=True, exist_ok=True)
    (root / "geom" / split).mkdir(parents=True, exist_ok=True)
    geom = rng.random((64, NF)).astype("float32")
    unsafe = geom[:, 0] < 0.2  # geometry carries the whole signal
    ttc = geom[:, 0] < 0.15
    dac = geom[:, 8] > 0.85
    nc = geom[:, 0] < 0.05
    np.savez(root / "data" / split / f"{name}.npz", scene_token=name,
             relative_trajectories=rng.normal(size=(64, 8, 3)).astype("float32"),
             reference_feasible=~(unsafe | ttc | dac | nc), reference_violation=(unsafe | dac).astype(int),
             score__time_to_collision_within_bound=1 - ttc.astype(int),
             score__drivable_area_compliance=1 - dac.astype(int),
             score__no_at_fault_collisions=1 - nc.astype(int))
    np.savez(root / "lat" / split / f"{name}.npz", eoa_latents=rng.normal(size=(64, 4, 4096)).astype("float32"))
    np.savez(root / "geom" / split / f"{name}.npz", geom_features=geom, feature_names=np.asarray(gf.FEATURE_NAMES),
             scene_token=np.asarray(name))


@unittest.skipUnless(HAS_TORCH, "PyTorch required")
class EndToEndTest(unittest.TestCase):
    def test_geometry_signal_is_recovered_and_latent_noise_is_not(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory(prefix="wm-baseline-") as tmp:
            root = Path(tmp)
            for i in range(16):
                write_scene(root, "train" if i < 12 else "val", f"dev{i:02d}", rng)
            for i in range(6):
                write_scene(root, "test", f"new{i:02d}", rng)
            # A fresh-split scene that is also in the development pool must be excluded.
            write_scene(root, "test", "dev00", rng)
            env = dict(
                os.environ, WM_ACTION_DATA_ROOT=str(root / "data"), WM_LATENT_ROOT=str(root / "lat"),
                WM_GEOM_ROOT=str(root / "geom"), WM_RISK_RANKER_OUTPUT=str(root / "oof"),
                WM_RISK_CV_FOLDS="2", WM_RISK_EPOCHS="25", WM_RISK_SEED="0", WM_RISK_LR="2e-3",
                WM_FRESH_DATA_ROOT=str(root / "data"), WM_FRESH_LATENT_ROOT=str(root / "lat"),
                WM_FRESH_GEOM_ROOT=str(root / "geom"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            )
            modes = ("action", "geom", "action_geom", "resid1", "action_geom_wm1")
            for mode in modes:
                for fold in range(2):
                    run = subprocess.run(
                        [sys.executable, "tools/rl_prep/train_risk_baselines.py"],
                        env=dict(env, WM_BASELINE_MODE=mode, WM_RISK_CV_FOLD=str(fold)),
                        capture_output=True, text=True, cwd=REPO)
                    if run.returncode:
                        raise RuntimeError(run.stdout[-3000:] + run.stderr[-3000:])
            metrics = json.loads((root / "oof/geom/seed_0/fold_0/metrics.json").read_text())
            self.assertEqual(metrics["fresh_scenes"], 6)  # dev00 excluded
            self.assertEqual(metrics["selection_protocol"], "inner_scene_holdout_v1")
            self.assertTrue(set(metrics["outer_scenes"]).isdisjoint(metrics["training_scenes"]))

            out = root / "summary.json"
            summ.main(["--oof-root", str(root / "oof"), "--modes", *modes, "--reference", "action",
                       "--folds", "2", "--resamples", "300", "--data-root", str(root / "data"),
                       "--geom-root", str(root / "geom"), "--output", str(out)])
            report = json.loads(out.read_text())
            for kind in ("oof", "fresh"):
                # ttc_failure is determined by geometry feature 0 alone.
                geom_auc = report[kind]["modes"]["geom"]["auroc"]["ttc_failure"]
                action_auc = report[kind]["modes"]["action"]["auroc"]["ttc_failure"]
                resid_auc = report[kind]["modes"]["resid1"]["auroc"]["ttc_failure"]
                self.assertGreater(geom_auc, 0.85, kind)
                self.assertLess(action_auc, 0.65, kind)
                self.assertLess(resid_auc, 0.65, kind)  # latents are pure noise here
            audit = report["geometric_feature_audit"]["features"]
            self.assertLess(audit["min_clearance"]["ttc_failure"]["within_scene_auroc"], 0.1)  # low value = risky
            self.assertTrue(out.with_suffix(".md").read_text().startswith("### Cross-fitted"))
            summ.main(["--audit-only", "--data-root", str(root / "data"), "--geom-root", str(root / "geom")])
            with self.assertRaises(SystemExit):  # audit needs both roots
                summ.main(["--audit-only", "--data-root", str(root / "data")])
            with self.assertRaises(SystemExit):  # never overwrite
                summ.main(["--oof-root", str(root / "oof"), "--modes", *modes, "--reference", "action",
                           "--folds", "2", "--output", str(out)])
            rerun = subprocess.run(
                [sys.executable, "tools/rl_prep/train_risk_baselines.py"],
                env=dict(env, WM_BASELINE_MODE="geom", WM_RISK_CV_FOLD="0"), capture_output=True, text=True, cwd=REPO)
            self.assertNotEqual(rerun.returncode, 0)  # refuses to overwrite an existing run


if __name__ == "__main__":
    unittest.main()
