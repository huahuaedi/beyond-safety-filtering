import json, tempfile, unittest
from pathlib import Path
import numpy as np
import build_margin_preferences as b
from offline_verified_data import absolute_from_relative

NAMES = ["min_clearance", "min_clearance_2s", "mean_clearance", "t_clear_lt_0m", "t_clear_lt_1m", "t_clear_lt_3m",
         "min_headway", "n_agents_lt_3m", "offroad_frac_max", "offroad_frac_mean", "offroad_frac_final", "t_first_offroad"]
M = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress", "comfort", "driving_direction_compliance")


def write_scene(root, groot, token, n=6):
    rel = np.zeros((n, 8, 3), np.float32)
    rel[:, :, 0] = np.linspace(1.0, 2.0, n)[:, None]               # candidate i advances faster as i grows
    d = dict(scene_token=token, split="train", relative_trajectories=rel, absolute_trajectories=absolute_from_relative(rel),
             expert_action_normalized=np.zeros((8, 3), np.float32))
    for k in M:
        d["score__" + k] = np.ones(n, np.float32)
        d["reference_probe__" + k] = np.ones(2, np.float32)
    d["score__ego_progress"] = np.linspace(0.5, 0.9, n).astype(np.float32)
    d["reference_probe__ego_progress"] = np.array([0.6, 0.6], np.float32)
    d["score__drivable_area_compliance"][4:] = 0.0                 # candidates 4, 5 leave the drivable area
    d["reference_probe_absolute_trajectories"] = np.stack([d["absolute_trajectories"][3]] * 2).astype(np.float32)   # reference output = candidate 3
    np.savez(root / "train" / f"{token}.npz", **d)
    g = np.zeros((n, 12))
    g[:, 0] = [1.0, 2.0, 5.0, 3.0, 0.5, 0.2]                        # min_clearance: candidate 2 has the largest margin
    g[4:, 8] = 0.5
    np.savez(groot / "train" / f"{token}.npz", geom_features=g, feature_names=np.array(NAMES))


class MarginPairTest(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); t = Path(self.t.name)
        (t / "data" / "train").mkdir(parents=True); (t / "geom" / "train").mkdir(parents=True)
        for i in range(5):
            write_scene(t / "data", t / "geom", f"s{i}")
        (t / "m.json").write_text(json.dumps({"schema_version": 2, "train": [{"token": f"s{i}", "split": "train"} for i in range(5)]}))
        (t / "n.json").write_text(json.dumps({"norm_stats": {"libero": {"q01": [0, -1, -1], "q99": [4, 1, 1]}}}))
        self.dir = t

    def tearDown(self):
        self.t.cleanup()

    def run_main(self, rule):
        import sys
        old = sys.argv
        sys.argv = ["x", "--root", str(self.dir / "data"), "--geom-root", str(self.dir / "geom"), "--manifest", str(self.dir / "m.json"),
                    "--norm-stats", str(self.dir / "n.json"), "--pass-rule", rule, "--output-dir", str(self.dir / f"out_{rule}")]
        try:
            b.main()
        finally:
            sys.argv = old
        return json.loads((self.dir / f"out_{rule}" / "report.json").read_text()), \
            [json.loads(l) for l in (self.dir / f"out_{rule}" / "pairs.jsonl").read_text().splitlines()]

    def test_simulator_rule_picks_largest_margin_winner_and_expert_closest_loser(self):
        r, pairs = self.run_main("simulator")
        self.assertEqual(r["scenes_with_margin_pair"], 5)
        self.assertTrue(all(p["winner"] == 2 for p in pairs))          # passing set {0..3}, largest clearance is index 2
        # violating set {4, 5}; with q01=0, q99=4 the zero expert action equals an advance of 2.0, which is candidate 5
        self.assertTrue(all(p["loser"] == 5 for p in pairs))

    def test_margin_winner_progress_differs_from_pdms_top2_labels(self):
        r, _ = self.run_main("simulator")
        w = r["winner_minus_reference"]["progress"]["mean_x100"]       # winner (candidate 2) progress 0.66 vs reference 0.60
        top2 = r["pdms_top2_positives_minus_reference"]["progress"]["mean_x100"]   # candidates 2 and 3: mean 0.70
        self.assertAlmostEqual(w, 6.0, places=3)
        self.assertAlmostEqual(top2, 10.0, places=3)
        self.assertGreater(top2, w)                                     # PDMS-ranked labels are faster than the margin winner

    def test_stratification_by_reference_safety(self):
        r, _ = self.run_main("simulator")
        st = r["stratified_by_reference_safety"]
        self.assertEqual(st["ref_safe"]["scenes"], 5)                  # reference probes are clean in the synthetic scenes
        self.assertEqual(st["ref_unsafe"]["scenes"], 0)
        self.assertEqual(st["ref_safe"]["scenes_with_margin_pair"], 5)
        self.assertAlmostEqual(st["ref_safe"]["margin_winner_progress"]["mean_x100"], 6.0, places=3)
        self.assertAlmostEqual(st["ref_safe"]["pdms_top2_progress"]["mean_x100"], 10.0, places=3)
        self.assertIsNone(st["ref_unsafe"]["margin_winner_progress"])

    def test_pdms_winner_rule_keeps_the_same_loser_and_is_faster(self):
        import sys
        old = sys.argv
        sys.argv = ["x", "--root", str(self.dir / "data"), "--geom-root", str(self.dir / "geom"), "--manifest", str(self.dir / "m.json"),
                    "--norm-stats", str(self.dir / "n.json"), "--winner-rule", "pdms", "--output-dir", str(self.dir / "out_pdms")]
        try:
            b.main()
        finally:
            sys.argv = old
        pdms_pairs = [json.loads(l) for l in (self.dir / "out_pdms" / "pairs.jsonl").read_text().splitlines()]
        _, margin_pairs = self.run_main("simulator")
        self.assertEqual([p["loser"] for p in pdms_pairs], [p["loser"] for p in margin_pairs])   # same loser
        self.assertTrue(all(p["winner"] == 3 for p in pdms_pairs))      # highest-PDMS passing candidate is the fastest passing one
        self.assertTrue(all(p["winner"] == 2 for p in margin_pairs))

    def test_nearest_rule_picks_the_passing_candidate_closest_to_the_reference_output(self):
        import sys
        old = sys.argv
        sys.argv = ["x", "--root", str(self.dir / "data"), "--geom-root", str(self.dir / "geom"), "--manifest", str(self.dir / "m.json"),
                    "--norm-stats", str(self.dir / "n.json"), "--winner-rule", "nearest", "--output-dir", str(self.dir / "out_nearest")]
        try:
            b.main()
        finally:
            sys.argv = old
        pairs = [json.loads(l) for l in (self.dir / "out_nearest" / "pairs.jsonl").read_text().splitlines()]
        rep = json.loads((self.dir / "out_nearest" / "report.json").read_text())
        self.assertTrue(all(p["winner"] == 3 for p in pairs))           # the reference output is candidate 3 itself
        self.assertAlmostEqual(rep["stratified_by_reference_safety"]["ref_safe"]["winner_deviation_from_reference_m"]["mean_x100"], 0.0, places=6)
        self.assertGreater(rep["stratified_by_reference_safety"]["ref_safe"]["pdms_top2_deviation_from_reference_m"]["mean_x100"], 0.0)

    def test_nearest_rule_requires_reference_geometry(self):
        rel = np.zeros((3, 8, 3), np.float32)
        sc = {k: np.ones(3) for k in M}; sc["drivable_area_compliance"] = np.array([1, 1, 0.0])
        geom = {n: np.zeros(3) for n in NAMES}
        with self.assertRaises(ValueError):
            b.scene_pair(rel, sc, {}, geom, np.zeros((8, 3), np.float32), np.array([0, -1, -1.]), np.array([4, 1, 1.]), "simulator", winner_rule="nearest")

    def test_geometric_rule_and_no_pair_when_everything_passes(self):
        r, pairs = self.run_main("geometric")
        self.assertEqual(r["scenes_with_margin_pair"], 5)
        rel = np.zeros((3, 8, 3), np.float32)
        sc = {k: np.ones(3) for k in M}
        geom = {n: np.zeros(3) for n in NAMES}
        self.assertIsNone(b.scene_pair(rel, sc, {}, geom, np.zeros((8, 3), np.float32), np.array([0, -1, -1.]), np.array([4, 1, 1.]), "simulator"))


if __name__ == "__main__":
    unittest.main()
