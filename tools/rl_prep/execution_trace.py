"""Desired-vs-executed trajectory traces from the NAVSIM PDM simulator.

NAVSIM's ``pdm_score()`` runs every candidate through an LQR tracker plus a
kinematic bicycle model (``PDMSimulator``) and scores the *executed* states,
but only returns scalar sub-scores. ``navsim_score_worker.py --trace-npz``
keeps those states; this module turns them into execution quantities:

- tracking error of the executed trajectory w.r.t. the desired one, measured
  at the same time index (so longitudinal error also captures speed lag);
- signed drivable-area margin of the four footprint corners (positive inside,
  negative outside), for both desired and executed states;
- executed actuation (steering angle/rate, longitudinal/lateral acceleration).

State arrays follow NAVSIM's ``StateIndex`` layout ([..., T, 11], global
frame, rear-axle reference point). The NAVSIM tracker ignores the desired
trajectory's velocity/acceleration and follows only interpolated poses, so the
tracking error reflects whether the 8 predicted poses are trackable by that
tracker, not by a production vehicle control stack.

Pure NumPy (shapely only for the map adapter), so it is testable without
nuPlan/NAVSIM.
"""
from __future__ import annotations

import numpy as np

# NAVSIM pdm_enums.StateIndex
X, Y, HEADING = 0, 1, 2
VELOCITY_X, VELOCITY_Y = 3, 4
ACCELERATION_X, ACCELERATION_Y = 5, 6
STEERING_ANGLE, STEERING_RATE = 7, 8
ANGULAR_VELOCITY, ANGULAR_ACCELERATION = 9, 10
STATE_SIZE = 11

TRACE_SCHEMA_VERSION = 1
DT = 0.1  # PDMSimulator proposal sampling: 40 poses @ 0.1 s (+ initial state)


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def tracking_errors(desired, executed):
    """Executed pose expressed in the desired pose's frame, per time index.

    Returns (longitudinal, lateral, heading) errors, each [..., T]. Positive
    longitudinal = executed ahead of desired; positive lateral = to the left.
    """
    desired = np.asarray(desired, dtype=np.float64)
    executed = np.asarray(executed, dtype=np.float64)
    dx = executed[..., X] - desired[..., X]
    dy = executed[..., Y] - desired[..., Y]
    cos, sin = np.cos(desired[..., HEADING]), np.sin(desired[..., HEADING])
    lon = cos * dx + sin * dy
    lat = -sin * dx + cos * dy
    heading = wrap_angle(executed[..., HEADING] - desired[..., HEADING])
    return lon, lat, heading


def fill_finite_difference_dynamics(states, dt=DT):
    """Give a pose-only desired trajectory body-frame velocities/yaw rate.

    NAVSIM's interpolated desired states carry zero velocity, which would make
    the scorer's collision classification (stopped vs. active ego) meaningless
    when scoring them open-loop. Index 0 (the logged initial ego state) is kept.
    """
    out = np.array(states, dtype=np.float64, copy=True)
    if out.shape[-2] < 2:
        return out
    dxy = np.diff(out[..., :, [X, Y]], axis=-2)  # [..., T-1, 2]
    heading = out[..., 1:, HEADING]
    cos, sin = np.cos(heading), np.sin(heading)
    out[..., 1:, VELOCITY_X] = (cos * dxy[..., 0] + sin * dxy[..., 1]) / dt
    out[..., 1:, VELOCITY_Y] = (-sin * dxy[..., 0] + cos * dxy[..., 1]) / dt
    out[..., 1:, ANGULAR_VELOCITY] = wrap_angle(np.diff(out[..., :, HEADING], axis=-1)) / dt
    out[..., 1:, ACCELERATION_X] = np.diff(out[..., :, VELOCITY_X], axis=-1) / dt
    out[..., 1:, ACCELERATION_Y] = np.diff(out[..., :, VELOCITY_Y], axis=-1) / dt
    out[..., 1:, ANGULAR_ACCELERATION] = np.diff(out[..., :, ANGULAR_VELOCITY], axis=-1) / dt
    return out


def footprint_corners(states, half_length, half_width, rear_axle_to_center):
    """[..., T, 4, 2] corners (FL, RL, RR, FR), same geometry as NAVSIM's
    ``state_array_to_coords_array``."""
    states = np.asarray(states, dtype=np.float64)
    heading = states[..., HEADING]
    cos, sin = np.cos(heading), np.sin(heading)
    cx = states[..., X] + rear_axle_to_center * cos
    cy = states[..., Y] + rear_axle_to_center * sin
    lon = np.array([half_length, -half_length, -half_length, half_length])
    lat = np.array([half_width, half_width, -half_width, -half_width])
    x = cx[..., None] + lon * cos[..., None] - lat * sin[..., None]
    y = cy[..., None] + lon * sin[..., None] + lat * cos[..., None]
    return np.stack([x, y], axis=-1)


def signed_corner_margin(corners, contains_xy, boundary_distance):
    """Minimum signed distance of the footprint corners to the drivable
    boundary, [..., T]. ``contains_xy(x, y) -> bool[N]`` and
    ``boundary_distance(points[N, 2]) -> float[N]`` are map callables."""
    corners = np.asarray(corners, dtype=np.float64)
    flat = corners.reshape(-1, 2)
    inside = np.asarray(contains_xy(flat[:, 0], flat[:, 1]), dtype=bool)
    dist = np.asarray(boundary_distance(flat), dtype=np.float64)
    signed = np.where(inside, dist, -dist).reshape(corners.shape[:-1])
    return signed.min(axis=-1)


def shapely_drivable_callables(polygons):
    """Map callables over the union of drivable polygons (shapely >= 2)."""
    import shapely

    polygons = [p for p in polygons if p is not None and not p.is_empty]
    if not polygons:
        return (lambda x, y: np.zeros(len(x), dtype=bool),
                lambda pts: np.full(len(pts), np.inf))
    union = shapely.union_all(polygons)
    boundary = union.boundary
    shapely.prepare(union)
    shapely.prepare(boundary)
    return (
        lambda x, y: shapely.contains_xy(union, x, y),
        lambda pts: shapely.distance(boundary, shapely.points(pts)),
    )


SCORE_KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
              "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score")
TRACE_SCALAR_KEYS = ("collision_time_idx", "ttc_time_idx", "progress_raw",
                     "desired__no_at_fault_collisions", "desired__drivable_area_compliance",
                     "desired__time_to_collision_within_bound")


def build_trace_payload(scene_token, meta_json, input_trajectories, vehicle_dims, rows, traces,
                        errors, contains_xy, boundary_distance, num_states=41):
    """Arrays saved per scoring call. ``traces[i]`` is the per-candidate
    dict from navsim_score_worker.pdm_score_with_trace (None for a row whose
    official scoring failed); ``rows[i]`` its official score dict;
    ``vehicle_dims`` = (half_length, half_width, rear_axle_to_center, wheel_base)."""
    G, T = len(traces), num_states
    nan_states = np.full((T, STATE_SIZE), np.nan)
    no_areas = np.zeros((T, 3), dtype=bool)

    def stack(key, fill):
        return np.stack([t[key] if t is not None else fill for t in traces])

    desired = stack("desired_states", nan_states).astype(np.float64)
    executed = stack("executed_states", nan_states).astype(np.float64)
    lon, lat, heading = tracking_errors(desired, executed)
    ok = np.array([t is not None for t in traces], dtype=bool)
    dims = tuple(float(x) for x in vehicle_dims[:3])
    desired_margin = np.full((G, T), np.nan)
    executed_margin = np.full((G, T), np.nan)
    if ok.any():
        desired_margin[ok] = signed_corner_margin(footprint_corners(desired[ok], *dims), contains_xy, boundary_distance)
        executed_margin[ok] = signed_corner_margin(footprint_corners(executed[ok], *dims), contains_xy, boundary_distance)
    reference = next((t["pdm_reference_executed_states"] for t in traces if t is not None), nan_states)

    payload = {
        "schema_version": np.asarray(TRACE_SCHEMA_VERSION, dtype=np.int32),
        "scene_token": np.asarray(scene_token),
        "meta_json": np.asarray(meta_json),
        "input_trajectories": np.asarray(input_trajectories, dtype=np.float32),
        "initial_ego_state": desired[ok][0, 0] if ok.any() else nan_states[0],
        "vehicle_dims": np.asarray(vehicle_dims, dtype=np.float64),
        "row_ok": ok,
        "row_error": np.asarray(errors, dtype=str),
        "open_loop_error": np.asarray([(t or {}).get("open_loop_error", "") for t in traces], dtype=str),
        "desired_states": desired.astype(np.float32),
        "executed_states": executed.astype(np.float32),
        "pdm_reference_executed_states": np.asarray(reference, dtype=np.float32),
        "tracking_lon_err": lon.astype(np.float32),
        "tracking_lat_err": lat.astype(np.float32),
        "tracking_heading_err": heading.astype(np.float32),
        "desired_margin": desired_margin.astype(np.float32),
        "executed_margin": executed_margin.astype(np.float32),
        "executed_ego_areas": stack("executed_ego_areas", no_areas).astype(bool),
        "desired_ego_areas": stack("desired_ego_areas", no_areas).astype(bool),
    }
    for key in TRACE_SCALAR_KEYS:
        payload[key] = np.asarray([t[key] if t is not None else np.nan for t in traces], dtype=np.float32)
    for key in SCORE_KEYS:
        payload[f"score__{key}"] = np.asarray([r[key] for r in rows], dtype=np.float32)
    return payload


def first_index(mask, axis=-1):
    """First True index along ``axis``; -1 where never True."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.argmax(mask, axis=axis)
    return np.where(mask.any(axis=axis), idx, -1)


SUMMARY_KEYS = (
    "max_abs_lon_err", "max_abs_lat_err", "max_abs_heading_err", "rms_pos_err",
    "final_pos_err", "executed_min_margin", "executed_min_margin_t",
    "desired_min_margin", "desired_min_margin_t", "executed_first_offroad_t",
    "desired_first_offroad_t", "max_abs_steering", "max_abs_steering_rate",
    "max_abs_long_acc", "max_abs_lat_acc", "max_speed",
)


def summarize(trace):
    """Per-row scalar summaries from a loaded trace (dict of arrays).

    Times are in seconds; ``*_first_offroad_t`` is NaN when never offroad
    (per the continuous margin, not the scorer flag).
    """
    lon, lat, head = trace["tracking_lon_err"], trace["tracking_lat_err"], trace["tracking_heading_err"]
    pos = np.hypot(lon, lat)
    executed = trace["executed_states"]
    ex_margin, de_margin = trace["executed_margin"], trace["desired_margin"]

    def first_t(margin):
        idx = first_index(margin < 0.0)
        return np.where(idx >= 0, idx * DT, np.nan)

    return {
        "max_abs_lon_err": np.abs(lon).max(-1),
        "max_abs_lat_err": np.abs(lat).max(-1),
        "max_abs_heading_err": np.abs(head).max(-1),
        "rms_pos_err": np.sqrt((pos ** 2).mean(-1)),
        "final_pos_err": pos[..., -1],
        "executed_min_margin": ex_margin.min(-1),
        "executed_min_margin_t": ex_margin.argmin(-1) * DT,
        "desired_min_margin": de_margin.min(-1),
        "desired_min_margin_t": de_margin.argmin(-1) * DT,
        "executed_first_offroad_t": first_t(ex_margin),
        "desired_first_offroad_t": first_t(de_margin),
        "max_abs_steering": np.abs(executed[..., STEERING_ANGLE]).max(-1),
        "max_abs_steering_rate": np.abs(executed[..., STEERING_RATE]).max(-1),
        "max_abs_long_acc": np.abs(executed[..., ACCELERATION_X]).max(-1),
        # Kinematic bicycle: lateral acceleration = v_x * yaw rate.
        "max_abs_lat_acc": np.abs(executed[..., VELOCITY_X] * executed[..., ANGULAR_VELOCITY]).max(-1),
        "max_speed": np.hypot(executed[..., VELOCITY_X], executed[..., VELOCITY_Y]).max(-1),
    }


# Failure attribution, per metric, comparing the desired trajectory scored
# open-loop (no tracker) with the executed one (official NAVSIM score).
ATTRIBUTION_LABELS = ("pass", "planned", "execution_induced", "execution_rescued")


def attribute(desired_pass, executed_pass):
    """planned: desired already fails and executed fails; execution_induced:
    desired passes but executed fails; execution_rescued: desired fails but
    executed passes."""
    desired_pass = np.asarray(desired_pass, dtype=bool)
    executed_pass = np.asarray(executed_pass, dtype=bool)
    out = np.full(desired_pass.shape, "pass", dtype=object)
    out[~desired_pass & ~executed_pass] = "planned"
    out[desired_pass & ~executed_pass] = "execution_induced"
    out[~desired_pass & executed_pass] = "execution_rescued"
    return out


def roc_auc(scores, labels):
    """AUC of ``scores`` for positive ``labels`` (ties averaged); NaN if a
    class is empty."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
