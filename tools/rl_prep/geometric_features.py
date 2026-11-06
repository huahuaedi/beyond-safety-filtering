"""Non-learned geometric risk features for candidate ego trajectories.

This is the "strong non-WM baseline" for the world-model risk diagnostic:
constant-velocity forecasts of the agents observed at t=0 plus the drivable
map, evaluated against each candidate trajectory.  It deliberately uses only
information a deployed perception stack could provide (current agent boxes and
velocities, map polygons).  It never reads the logged future of other agents,
so it is not an oracle for the NAVSIM labels, which are computed from that
logged future.

The core is pure NumPy so it can be unit-tested without nuPlan/NAVSIM.  The
metric-cache adapter lives in ``extract_geometric_features.py``.

All returned features are bounded and roughly O(1), so no per-feature
standardisation is needed downstream.
"""
from __future__ import annotations

import numpy as np

# Candidate trajectories: 8 poses at 0.5 s spacing (see navsim_score_worker.py).
HORIZON_TIMES = 0.5 * np.arange(1, 9)
CLEARANCE_CAP = 30.0
NO_EVENT_TIME = 4.5  # sentinel "never happens within the 4 s horizon"
N_CIRCLES = 3

FEATURE_NAMES = (
    "min_clearance",
    "min_clearance_2s",
    "mean_clearance",
    "t_clear_lt_0m",
    "t_clear_lt_1m",
    "t_clear_lt_3m",
    "min_headway",
    "n_agents_lt_3m",
    "offroad_frac_max",
    "offroad_frac_mean",
    "offroad_frac_final",
    "t_first_offroad",
)


def local_to_global_poses(poses, origin):
    """Transform (x, y, heading) poses from the frame of ``origin`` to global."""
    poses = np.asarray(poses, dtype=np.float64)
    ox, oy, oh = origin
    c, s = np.cos(oh), np.sin(oh)
    x = c * poses[..., 0] - s * poses[..., 1] + ox
    y = s * poses[..., 0] + c * poses[..., 1] + oy
    return np.stack([x, y, poses[..., 2] + oh], axis=-1)


def _unit(heading):
    return np.stack([np.cos(heading), np.sin(heading)], axis=-1)


def ego_circles(rear_axle_poses, length, width, center_offset, n_circles=N_CIRCLES):
    """Circle cover of the ego footprint.

    ``rear_axle_poses`` is [..., 3] (x, y, heading).  Returns circles
    [..., n_circles, 2] and the shared radius.
    """
    poses = np.asarray(rear_axle_poses, dtype=np.float64)
    u = _unit(poses[..., 2])
    centers = poses[..., :2] + center_offset * u
    radius = width / 2.0
    half = max(length / 2.0 - radius, 0.0)
    offsets = np.linspace(-half, half, n_circles)
    return centers[..., None, :] + offsets[:, None] * u[..., None, :], radius


def agent_circles(pos, vel, heading, length, width, times, n_circles=N_CIRCLES):
    """Constant-velocity, constant-heading circle cover of each agent.

    Returns circles [N, T, n_circles, 2] and radii [N].
    """
    pos, vel = np.asarray(pos, np.float64), np.asarray(vel, np.float64)
    heading = np.asarray(heading, np.float64)
    radius = np.asarray(width, np.float64) / 2.0
    half = np.maximum(np.asarray(length, np.float64) / 2.0 - radius, 0.0)
    offsets = half[:, None] * np.linspace(-1.0, 1.0, n_circles)[None, :]  # [N, n]
    centers = pos[:, None, :] + vel[:, None, :] * np.asarray(times)[None, :, None]  # [N, T, 2]
    u = _unit(heading)  # [N, 2]
    circles = centers[:, :, None, :] + offsets[:, None, :, None] * u[:, None, None, :]
    return circles, radius


def _first_time_below(values, threshold, times):
    """First time at which values[c, t] < threshold, else NO_EVENT_TIME."""
    below = values < threshold
    first = np.where(below.any(axis=1), below.argmax(axis=1), -1)
    return np.where(first >= 0, np.asarray(times)[np.maximum(first, 0)], NO_EVENT_TIME)


def ego_speed(rear_axle_poses_local, times):
    """Speed estimate [C, T] from consecutive candidate poses (origin at t=0)."""
    poses = np.asarray(rear_axle_poses_local, dtype=np.float64)
    xy = np.concatenate([np.zeros_like(poses[:, :1, :2]), poses[:, :, :2]], axis=1)
    dt = np.diff(np.concatenate([[0.0], np.asarray(times, np.float64)]))
    return np.linalg.norm(np.diff(xy, axis=1), axis=-1) / dt[None, :]


def agent_clearance(ego_global_poses, agents, ego_dims, times=HORIZON_TIMES):
    """Per-candidate, per-time minimum clearance to any forecast agent: [C, T].

    Clearance is the circle-cover distance between footprints (negative means
    overlap).  With no agents it equals ``CLEARANCE_CAP``.  Also returns the
    per-agent minimum over time [C, N] for proximity counting.
    """
    num_candidates, num_times = ego_global_poses.shape[:2]
    if agents is None or len(agents["pos"]) == 0:
        return (
            np.full((num_candidates, num_times), CLEARANCE_CAP),
            np.full((num_candidates, 0), CLEARANCE_CAP),
        )
    ego, ego_radius = ego_circles(
        ego_global_poses, ego_dims["length"], ego_dims["width"], ego_dims["center_offset"]
    )  # [C, T, ne, 2]
    other, other_radius = agent_circles(
        agents["pos"], agents["vel"], agents["heading"], agents["length"], agents["width"], times
    )  # [N, T, na, 2]
    delta = ego[:, None, :, :, None, :] - other[None, :, :, None, :, :]  # [C, N, T, ne, na, 2]
    distance = np.linalg.norm(delta, axis=-1).min(axis=(-2, -1))  # [C, N, T]
    clearance = distance - ego_radius - other_radius[None, :, None]
    return (
        np.minimum(clearance.min(axis=1), CLEARANCE_CAP),
        np.minimum(clearance.min(axis=2), CLEARANCE_CAP),
    )


def footprint_points(rear_axle_poses, length, width, center_offset):
    """Nine boundary/centre sample points of the footprint: [..., 9, 2]."""
    poses = np.asarray(rear_axle_poses, dtype=np.float64)
    u = _unit(poses[..., 2])
    n = np.stack([-u[..., 1], u[..., 0]], axis=-1)
    center = poses[..., :2] + center_offset * u
    lon = np.array([1, 1, -1, -1, 1, -1, 0, 0, 0], dtype=np.float64) * length / 2.0
    lat = np.array([1, -1, 1, -1, 0, 0, 1, -1, 0], dtype=np.float64) * width / 2.0
    return center[..., None, :] + lon[:, None] * u[..., None, :] + lat[:, None] * n[..., None, :]


def scene_geometric_features(
    candidates_local, ego_origin_global, agents, ego_dims, inside_drivable, times=HORIZON_TIMES
):
    """Feature matrix [C, len(FEATURE_NAMES)] for one scene.

    ``candidates_local``: [C, 8, 3] rear-axle poses in the initial ego frame
    (the format handed to the NAVSIM scorer).  ``agents``: dict of global-frame
    arrays ``pos [N,2] vel [N,2] heading [N] length [N] width [N]`` or None.
    ``inside_drivable``: callable mapping points [..., 2] to a bool array of
    the same leading shape (True = on drivable area).
    """
    candidates_local = np.asarray(candidates_local, dtype=np.float64)
    if candidates_local.ndim != 3 or candidates_local.shape[-1] != 3:
        raise ValueError(f"expected [C, T, 3] candidates, got {candidates_local.shape}")
    if candidates_local.shape[1] != len(times):
        raise ValueError(f"candidate horizon {candidates_local.shape[1]} != {len(times)} times")
    if not np.isfinite(candidates_local).all():
        raise ValueError("non-finite candidate poses")
    times = np.asarray(times, dtype=np.float64)

    global_poses = local_to_global_poses(candidates_local, ego_origin_global)
    clearance_t, clearance_agent = agent_clearance(global_poses, agents, ego_dims, times)
    speed = ego_speed(candidates_local, times)

    points = footprint_points(
        global_poses, ego_dims["length"], ego_dims["width"], ego_dims["center_offset"]
    )
    inside = np.asarray(inside_drivable(points), dtype=bool)
    if inside.shape != points.shape[:-1]:
        raise ValueError(f"inside_drivable returned {inside.shape}, expected {points.shape[:-1]}")
    off_frac = 1.0 - inside.mean(axis=-1)  # [C, T]

    headway = (clearance_t / np.maximum(speed, 0.5)).min(axis=1)
    columns = [
        clearance_t.min(axis=1) / 10.0,
        clearance_t[:, : max(len(times) // 2, 1)].min(axis=1) / 10.0,
        clearance_t.mean(axis=1) / 10.0,
        _first_time_below(clearance_t, 0.0, times) / NO_EVENT_TIME,
        _first_time_below(clearance_t, 1.0, times) / NO_EVENT_TIME,
        _first_time_below(clearance_t, 3.0, times) / NO_EVENT_TIME,
        np.clip(headway, -5.0, 20.0) / 10.0,
        (clearance_agent < 3.0).sum(axis=1) / 10.0,
        off_frac.max(axis=1),
        off_frac.mean(axis=1),
        off_frac[:, -1],
        _first_time_below(1.0 - off_frac, 1.0 - 1e-9, times) / NO_EVENT_TIME,
    ]
    features = np.stack(columns, axis=1).astype(np.float32)
    if features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise RuntimeError("geometric feature construction produced an invalid matrix")
    return features


# ---- TTC-aware features -------------------------------------------------------------------
# NAVSIM's TTC term projects the ego forward at constant velocity for a short window from
# every trajectory step and fails the step if the projected footprint hits an agent.  The 12
# features above only look at the pose on the 0.5 s grid, so they cannot see a candidate that
# is safe on the grid but runs into a slower lead agent a fraction of a second later.
TTC_WINDOW = (0.25, 0.5, 0.75, 1.0)
MOVING_SPEED = 0.1  # m/s: NAVSIM ignores TTC while the ego is (almost) stationary

TTC_FEATURE_NAMES = (
    "ttc_violation_frac",
    "ttc_first_violation",
    "ttc_min_proj_clearance",
    "ttc_min_proj_clearance_early",
    "ttc_violation_depth",
    "clearance_closing_rate",
)


def scene_ttc_features(candidates_local, ego_origin_global, agents, ego_dims, times=HORIZON_TIMES):
    """TTC-style features [C, len(TTC_FEATURE_NAMES)] for one scene (same inputs as above)."""
    candidates_local = np.asarray(candidates_local, dtype=np.float64)
    if candidates_local.ndim != 3 or candidates_local.shape[-1] != 3:
        raise ValueError(f"expected [C, T, 3] candidates, got {candidates_local.shape}")
    if not np.isfinite(candidates_local).all():
        raise ValueError("non-finite candidate poses")
    times = np.asarray(times, dtype=np.float64)
    window = np.asarray(TTC_WINDOW, dtype=np.float64)
    num_c, num_t = candidates_local.shape[:2]
    global_poses = local_to_global_poses(candidates_local, ego_origin_global)
    speed = ego_speed(candidates_local, times)  # [C, T]

    # ego pose at (t_i + dt) under constant velocity along the pose heading
    heading_u = _unit(global_poses[..., 2])  # [C, T, 2]
    shift = speed[:, :, None, None] * window[None, None, :, None] * heading_u[:, :, None, :]
    proj = np.broadcast_to(global_poses[:, :, None, :], (num_c, num_t, len(window), 3)).copy()
    proj[..., :2] += shift
    proj_times = (times[:, None] + window[None, :]).reshape(-1)
    clearance, _ = agent_clearance(proj.reshape(num_c, -1, 3), agents, ego_dims, proj_times)
    clearance = clearance.reshape(num_c, num_t, len(window))

    moving = speed > MOVING_SPEED  # [C, T]
    hit = (clearance < 0.0).any(axis=2) & moving  # [C, T]
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    first_time = np.where(first >= 0, times[np.maximum(first, 0)], NO_EVENT_TIME)
    masked = np.where(moving[:, :, None], clearance, CLEARANCE_CAP)
    depth = np.clip(-masked.min(axis=2), 0.0, 5.0).max(axis=1)
    grid, _ = agent_clearance(global_poses, agents, ego_dims, times)
    early = max(num_t // 2, 1)
    closing = (grid[:, 0] - grid[:, min(3, num_t - 1)]) / (times[min(3, num_t - 1)] - times[0])
    columns = [
        hit.mean(axis=1),
        first_time / NO_EVENT_TIME,
        masked.min(axis=(1, 2)) / 10.0,
        masked[:, :early].min(axis=(1, 2)) / 10.0,
        depth / 5.0,
        np.clip(closing, -10.0, 10.0) / 10.0,
    ]
    features = np.stack(columns, axis=1).astype(np.float32)
    if features.shape[1] != len(TTC_FEATURE_NAMES) or not np.isfinite(features).all():
        raise RuntimeError("TTC feature construction produced an invalid matrix")
    return features
