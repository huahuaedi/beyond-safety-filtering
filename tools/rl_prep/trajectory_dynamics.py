"""Kinematic diagnostics for future SE(2) pose trajectory chunks.

This is an auditable feasibility proxy, not a high-fidelity tire model.  The
first predicted pose is preceded by the current ego pose (0, 0, 0).  Because
the collection files do not store current ego speed, acceleration starts at
the second predicted interval and jerk starts one interval later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicsLimits:
    max_speed: float = 20.0
    max_acceleration: float = 4.0
    max_deceleration: float = 6.0
    max_abs_jerk: float = 8.0
    max_abs_curvature: float = 0.25
    max_abs_curvature_rate: float = 0.20
    max_lateral_acceleration: float = 4.0
    # Tire-road friction budget.  Longitudinal and lateral acceleration share
    # this budget through sqrt(a_lon^2 + a_lat^2) <= mu*g; they are not two
    # independent boxes. Values may also be length-N arrays for per-scene or
    # per-candidate limits.
    friction_coefficient: float = 0.90
    gravity: float = 9.81
    min_motion: float = 1e-3

    def to_dict(self):
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(self).items()
        }


def _max_abs(values, axis=1):
    return np.max(np.abs(values), axis=axis) if values.shape[axis] else np.zeros(values.shape[0])


def _per_candidate(value, n: int, name: str):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(n, float(array), dtype=np.float64)
    if array.shape != (n,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"{name} must be a positive scalar or shape ({n},), got {array.shape}")
    return array


def analyze_pose_trajectories(trajectories, dt: float, limits=None):
    """Analyze accumulated poses shaped [candidates, horizon, (x,y,heading)]."""
    limits = limits or DynamicsLimits()
    trajectories = np.asarray(trajectories, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[2] != 3:
        raise ValueError(f"expected [N,H,3] trajectories, got {trajectories.shape}")
    if trajectories.shape[1] < 3 or dt <= 0 or not np.isfinite(trajectories).all():
        raise ValueError("trajectories must be finite with H>=3 and dt>0")

    n = trajectories.shape[0]
    limit = {
        name: _per_candidate(getattr(limits, name), n, name)
        for name in (
            "max_speed", "max_acceleration", "max_deceleration", "max_abs_jerk",
            "max_abs_curvature", "max_abs_curvature_rate", "friction_coefficient",
            "gravity", "min_motion",
        )
    }
    origin = np.zeros((n, 1, 3), dtype=np.float64)
    poses = np.concatenate([origin, trajectories], axis=1)
    xy_step = np.diff(poses[:, :, :2], axis=1)
    distance = np.linalg.norm(xy_step, axis=2)
    speed = distance / dt
    acceleration = np.diff(speed, axis=1) / dt
    jerk = np.diff(acceleration, axis=1) / dt

    heading = np.unwrap(poses[:, :, 2], axis=1)
    heading_step = np.diff(heading, axis=1)
    curvature = np.divide(
        heading_step,
        distance,
        out=np.zeros_like(heading_step),
        where=distance > limit["min_motion"][:, None],
    )
    curvature_rate = np.diff(curvature, axis=1) / dt
    lateral_acceleration = speed * speed * curvature
    # Acceleration is defined between consecutive speed samples, so pair it
    # with the lateral acceleration at the later sample of the same interval.
    friction_acceleration = np.sqrt(
        acceleration * acceleration + lateral_acceleration[:, 1:] ** 2
    )
    friction_limit = limit["friction_coefficient"] * limit["gravity"]

    max_speed = np.max(speed, axis=1)
    max_acceleration = np.max(acceleration, axis=1)
    min_acceleration = np.min(acceleration, axis=1)
    max_abs_jerk = _max_abs(jerk)
    max_abs_curvature = _max_abs(curvature)
    max_abs_curvature_rate = _max_abs(curvature_rate)
    max_abs_lateral_acceleration = _max_abs(lateral_acceleration)
    max_friction_acceleration = np.max(friction_acceleration, axis=1)

    violations = np.stack(
        [
            np.maximum(0.0, max_speed / limit["max_speed"] - 1.0),
            np.maximum(0.0, max_acceleration / limit["max_acceleration"] - 1.0),
            np.maximum(0.0, -min_acceleration / limit["max_deceleration"] - 1.0),
            np.maximum(0.0, max_abs_jerk / limit["max_abs_jerk"] - 1.0),
            np.maximum(0.0, max_abs_curvature / limit["max_abs_curvature"] - 1.0),
            np.maximum(0.0, max_abs_curvature_rate / limit["max_abs_curvature_rate"] - 1.0),
            np.maximum(0.0, max_friction_acceleration / friction_limit - 1.0),
        ],
        axis=1,
    )
    feasible = np.all(violations <= 0.0, axis=1)
    return {
        "speed": speed.astype(np.float32),
        "acceleration": acceleration.astype(np.float32),
        "jerk": jerk.astype(np.float32),
        "curvature": curvature.astype(np.float32),
        "curvature_rate": curvature_rate.astype(np.float32),
        "lateral_acceleration": lateral_acceleration.astype(np.float32),
        "friction_acceleration": friction_acceleration.astype(np.float32),
        "friction_limit": friction_limit.astype(np.float32),
        "max_speed": max_speed.astype(np.float32),
        "max_acceleration": max_acceleration.astype(np.float32),
        "min_acceleration": min_acceleration.astype(np.float32),
        "max_abs_jerk": max_abs_jerk.astype(np.float32),
        "max_abs_curvature": max_abs_curvature.astype(np.float32),
        "max_abs_curvature_rate": max_abs_curvature_rate.astype(np.float32),
        "max_abs_lateral_acceleration": max_abs_lateral_acceleration.astype(np.float32),
        "max_friction_acceleration": max_friction_acceleration.astype(np.float32),
        "violation_components": violations.astype(np.float32),
        "violation_severity": violations.sum(axis=1).astype(np.float32),
        "dynamics_feasible": feasible.astype(np.bool_),
    }


# Backward-compatible name for callers/tests written before the distinction
# between incremental DriveVLA actions and accumulated future poses was made.
analyze_relative_trajectories = analyze_pose_trajectories
