import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_execution_traces as aet
import execution_trace as et

T = 41
DIMS = (2.5, 1.0, 1.5, 3.0)  # half_length, half_width, rear_axle_to_center, wheel_base


def states_from_xyh(x, y, h, speed=0.0):
    s = np.zeros((len(x), et.STATE_SIZE))
    s[:, et.X], s[:, et.Y], s[:, et.HEADING] = x, y, h
    s[:, et.VELOCITY_X] = speed
    return s


def straight(speed=5.0, y=0.0):
    t = et.DT * np.arange(T)
    return states_from_xyh(speed * t, np.full(T, y), np.zeros(T), speed)


def strip_callables(y_min=-4.0, y_max=4.0):
    """Infinite straight road |y| within [y_min, y_max]."""
    contains = lambda x, y: (np.asarray(y) > y_min) & (np.asarray(y) < y_max)
    distance = lambda pts: np.minimum(np.abs(pts[:, 1] - y_min), np.abs(pts[:, 1] - y_max))
    return contains, distance


def trace(desired, executed, desired_dac=1.0, desired_nc=1.0):
    return {
        "desired_states": desired, "executed_states": executed,
        "pdm_reference_executed_states": straight(),
        "executed_ego_areas": np.zeros((T, 3), bool), "desired_ego_areas": np.zeros((T, 3), bool),
        "collision_time_idx": np.inf, "ttc_time_idx": np.inf, "progress_raw": 10.0,
        "desired__no_at_fault_collisions": desired_nc, "desired__drivable_area_compliance": desired_dac,
        "desired__time_to_collision_within_bound": 1.0,
    }


def score_row(dac=1.0, nc=1.0):
    return {"no_at_fault_collisions": nc, "drivable_area_compliance": dac, "ego_progress": 1.0,
            "time_to_collision_within_bound": 1.0, "comfort": 1.0,
            "driving_direction_compliance": 1.0, "score": dac * nc}


class GeometryTest(unittest.TestCase):
    def test_tracking_error_is_in_desired_frame(self):
        desired = states_from_xyh([0.0], [0.0], [np.pi / 2])
        executed = states_from_xyh([1.0], [2.0], [np.pi / 2 + 0.1])
        lon, lat, head = et.tracking_errors(desired, executed)
        np.testing.assert_allclose([lon[0], lat[0], head[0]], [2.0, -1.0, 0.1], atol=1e-12)

    def test_heading_error_wraps(self):
        _, _, head = et.tracking_errors(states_from_xyh([0], [0], [np.pi - 0.05]),
                                        states_from_xyh([0], [0], [-np.pi + 0.05]))
        np.testing.assert_allclose(head, [0.1], atol=1e-12)

    def test_footprint_corners_match_navsim_layout(self):
        corners = et.footprint_corners(states_from_xyh([0.0], [0.0], [0.0]), 2.5, 1.0, 1.5)[0]
        # FL, RL, RR, FR around center (1.5, 0)
        np.testing.assert_allclose(corners, [[4.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [4.0, -1.0]])
        rotated = et.footprint_corners(states_from_xyh([0.0], [0.0], [np.pi / 2]), 2.5, 1.0, 1.5)[0]
        np.testing.assert_allclose(rotated[0], [-1.0, 4.0], atol=1e-12)

    def test_signed_margin_inside_and_outside(self):
        contains, distance = strip_callables()
        inside = et.signed_corner_margin(et.footprint_corners(straight(y=0.0), *DIMS[:3]), contains, distance)
        np.testing.assert_allclose(inside, 3.0)  # corner at |y|=1, boundary at 4
        outside = et.signed_corner_margin(et.footprint_corners(straight(y=3.5), *DIMS[:3]), contains, distance)
        np.testing.assert_allclose(outside, -0.5)

    def test_shapely_callables_agree_with_strip(self):
        try:
            import shapely.geometry as sg
        except ImportError:
            self.skipTest("shapely not installed")
        # Two touching polygons: union must remove the shared edge.
        contains, distance = et.shapely_drivable_callables(
            [sg.box(-100, -4, 0, 4), sg.box(0, -4, 100, 4)]
        )
        margin = et.signed_corner_margin(et.footprint_corners(straight(y=0.0), *DIMS[:3]), contains, distance)
        np.testing.assert_allclose(margin, 3.0, atol=1e-9)
        margin = et.signed_corner_margin(et.footprint_corners(straight(y=3.5), *DIMS[:3]), contains, distance)
        np.testing.assert_allclose(margin, -0.5, atol=1e-9)

    def test_finite_difference_dynamics_keeps_initial_state(self):
        s = straight(speed=5.0)
        s[1:, et.VELOCITY_X] = 0.0  # NAVSIM desired states carry zero velocity
        s[0, et.VELOCITY_X] = 4.2
        out = et.fill_finite_difference_dynamics(s)
        self.assertEqual(out[0, et.VELOCITY_X], 4.2)
        np.testing.assert_allclose(out[1:, et.VELOCITY_X], 5.0)
        np.testing.assert_allclose(out[1:, et.VELOCITY_Y], 0.0, atol=1e-12)
        np.testing.assert_allclose(out[1:, et.ANGULAR_VELOCITY], 0.0, atol=1e-12)

    def test_attribution_and_auc(self):
        labels = et.attribute([True, False, True, False], [True, False, False, True])
        self.assertEqual(list(labels), ["pass", "planned", "execution_induced", "execution_rescued"])
        self.assertEqual(et.roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(et.roc_auc([0.5, 0.5], [0, 1]), 0.5)
        self.assertTrue(np.isnan(et.roc_auc([0.1, 0.2], [1, 1])))


class PayloadAndAnalysisTest(unittest.TestCase):
    def make_payload(self, scene, specs, meta):
        """specs: list of (desired_y, executed_y, desired_dac, executed_dac) or None for an error row."""
        contains, distance = strip_callables()
        traces, rows, errors = [], [], []
        for spec in specs:
            if spec is None:
                traces.append(None); rows.append(score_row(0.0, 0.0)); errors.append("boom")
                continue
            dy, ey, ddac, edac = spec
            desired, executed = straight(y=dy), straight(y=ey)
            executed[:, et.STEERING_ANGLE] = 0.1 * ey
            t = trace(desired, executed, desired_dac=ddac)
            t["executed_ego_areas"][:, 1] = edac < 1.0
            traces.append(t); rows.append(score_row(edac)); errors.append("")
        return et.build_trace_payload(scene, json.dumps(meta), np.zeros((len(specs), 8, 3)), DIMS,
                                      rows, traces, errors, contains, distance, num_states=T)

    def test_payload_contents(self):
        payload = self.make_payload("s0", [(0.0, 0.5, 1.0, 1.0), None], {"step": 0})
        self.assertEqual(payload["desired_states"].shape, (2, T, et.STATE_SIZE))
        self.assertEqual(list(payload["row_ok"]), [True, False])
        np.testing.assert_allclose(payload["tracking_lat_err"][0], 0.5, atol=1e-6)
        np.testing.assert_allclose(payload["desired_margin"][0], 3.0, atol=1e-6)
        np.testing.assert_allclose(payload["executed_margin"][0], 2.5, atol=1e-6)
        self.assertTrue(np.isnan(payload["executed_margin"][1]).all())
        summary = et.summarize(payload)
        self.assertAlmostEqual(float(summary["max_abs_lat_err"][0]), 0.5, places=5)
        self.assertTrue(np.isnan(summary["executed_first_offroad_t"][0]))

    def test_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            kinds = ["candidate"] * 4 + ["current_probe"]
            specs_a = [(0.0, 0.0, 1.0, 1.0),   # pass
                       (0.0, 3.5, 1.0, 0.0),   # execution induced (tracker pushes out)
                       (3.5, 3.5, 0.0, 0.0),   # planned
                       (3.5, 0.0, 0.0, 1.0),   # rescued
                       (0.0, 0.2, 1.0, 1.0)]
            specs_b = [(0.0, 0.0, 1.0, 1.0)] * 4 + [None]
            for scene, specs in (("a", specs_a), ("b", specs_b)):
                payload = self.make_payload(scene, specs, {"step": 0, "row_kinds": kinds})
                np.savez_compressed(Path(tmp) / f"{scene}_step000000.npz", **payload)
            out = Path(tmp) / "out"
            argv = sys.argv
            sys.argv = ["x", "--trace-dir", tmp, "--output-dir", str(out)]
            try:
                aet.main()
            finally:
                sys.argv = argv
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["n_files"], 2)
            self.assertEqual(report["sanity"]["n_row_errors"], 1)
            self.assertEqual(report["sanity"]["scorer_flag_vs_dac_agreement"], 1.0)
            self.assertEqual(report["sanity"]["margin_sign_vs_dac_agreement"], 1.0)
            dac = report["attribution"]["candidate"]["dac"]
            self.assertEqual(dac["counts"], {"pass": 5, "planned": 1, "execution_induced": 1, "execution_rescued": 1})
            self.assertAlmostEqual(dac["execution_induced_share_of_failures"], 0.5)
            self.assertEqual(dac["scenes_with_execution_induced_failure"], 1)
            cont = report["continuous_candidates"]
            self.assertEqual(cont["n_execution_induced_dac"], 1)
            self.assertEqual(cont["auc_execution_induced_dac_given_desired_pass"]["max_abs_lat_err"], 1.0)
            self.assertEqual(cont["all_pass_groups"]["n"], 1)
            lines = (out / "rows.csv").read_text().strip().splitlines()
            self.assertEqual(len(lines), 1 + 10)


class RLTraceArgsTest(unittest.TestCase):
    def load(self, trace_dir, verify=False):
        import ast
        source = Path(__file__).with_name("rl_stage1_prototype.py").read_text()
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "execution_trace_args")
        scope = dict(EXECUTION_TRACE_DIR=trace_dir, EXECUTION_TRACE_VERIFY=verify, SEED=3,
                     EMU_HUB="/ckpt", CONTROL_ADAPTER_CHECKPOINT="", UNIFIED_CORRECTION_CHECKPOINT="",
                     Path=Path, json=json)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<trace-args>", "exec"), scope)
        return scope["execution_trace_args"]

    def test_disabled_by_default(self):
        self.assertEqual(self.load("")("tok", 2, {"step": 1, "row_kinds": ["candidate"] * 2}), [])
        self.assertEqual(self.load("/traces")("tok", 2, None), [])

    def test_enabled_args(self):
        args = self.load("/traces", verify=True)("tok", 2, {"step": 7, "row_kinds": ["candidate", "current_probe"]})
        self.assertEqual(args[:2], ["--trace-npz", "/traces/tok_step000007.npz"])
        self.assertEqual(json.loads(args[3])["seed"], 3)
        self.assertEqual(args[-1], "--verify-trace-scores")
        with self.assertRaises(ValueError):
            self.load("/traces")("tok", 3, {"step": 7, "row_kinds": ["candidate"]})


if __name__ == "__main__":
    unittest.main()
