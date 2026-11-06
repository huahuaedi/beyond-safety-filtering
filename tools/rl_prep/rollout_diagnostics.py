"""Pure-numpy diagnostics for group-relative continuous trajectory rollouts.

These statistics are observational only: they must never affect sampling or
optimization.  In particular, normalized advantage variance is not treated as
a proxy for policy entropy; diversity is measured directly in trajectory space.
"""

from __future__ import annotations

import numpy as np


def compute_rollout_diagnostics(
    abs_trajs: np.ndarray,
    sub_scores: dict[str, np.ndarray],
    advantages: np.ndarray,
    violations: np.ndarray,
    is_safe: np.ndarray,
    *,
    progress_values: np.ndarray | None = None,
    trajectory_epsilon: float = 0.05,
    score_epsilon: float = 1e-4,
) -> dict[str, float | int]:
    """Return group-level diversity and raw learning-signal diagnostics.

    ``trajectory_pairwise_rmse`` is the mean pairwise XY RMSE in metres.
    The zero-signal flag is deliberately conservative: trajectories, progress,
    and constraint-violation magnitudes must all be nearly indistinguishable.
    """
    trajs = np.asarray(abs_trajs, dtype=np.float64)
    if trajs.ndim != 3 or trajs.shape[-1] < 2:
        raise ValueError(f"abs_trajs must have shape [G,T,D>=2], got {trajs.shape}")
    group_size = trajs.shape[0]
    if group_size < 1:
        raise ValueError("abs_trajs must contain at least one candidate")

    xy = trajs[..., :2]
    if group_size > 1:
        delta = xy[:, None] - xy[None, :]
        pairwise_rmse = np.sqrt(np.mean(np.square(delta), axis=(-1, -2)))
        upper = pairwise_rmse[np.triu_indices(group_size, k=1)]
        pairwise_mean = float(upper.mean())
        pairwise_min = float(upper.min())
    else:
        pairwise_mean = pairwise_min = 0.0

    trajectory_xy_std = float(np.sqrt(np.mean(np.var(xy, axis=0))))
    terminal_xy_std = float(np.sqrt(np.mean(np.var(xy[:, -1], axis=0))))

    progress = np.asarray(
        sub_scores["ego_progress"] if progress_values is None else progress_values,
        dtype=np.float64,
    )
    if progress_values is None and "driving_direction_compliance" in sub_scores:
        progress = 0.5 * (
            progress
            + np.asarray(sub_scores["driving_direction_compliance"], dtype=np.float64)
        )
    nc = np.asarray(sub_scores["no_at_fault_collisions"], dtype=np.float64)
    dac = np.asarray(sub_scores["drivable_area_compliance"], dtype=np.float64)
    ttc = np.asarray(sub_scores["time_to_collision_within_bound"], dtype=np.float64)
    violation = np.asarray(violations, dtype=np.float64)
    advantage = np.asarray(advantages, dtype=np.float64)
    safe = np.asarray(is_safe, dtype=bool)
    for name, values in {
        "progress": progress,
        "nc": nc,
        "dac": dac,
        "ttc": ttc,
        "violations": violation,
        "advantages": advantage,
        "is_safe": safe,
    }.items():
        if values.shape != (group_size,):
            raise ValueError(f"{name} must have shape ({group_size},), got {values.shape}")

    progress_std = float(progress.std())
    violation_std = float(violation.std())
    zero_signal = int(
        pairwise_mean <= trajectory_epsilon
        and progress_std <= score_epsilon
        and violation_std <= score_epsilon
    )
    return {
        "trajectory_pairwise_rmse": pairwise_mean,
        "trajectory_pairwise_min_rmse": pairwise_min,
        "trajectory_xy_std": trajectory_xy_std,
        "terminal_xy_std": terminal_xy_std,
        "progress_raw_std": progress_std,
        "nc_raw_std": float(nc.std()),
        "dac_raw_std": float(dac.std()),
        "ttc_raw_std": float(ttc.std()),
        "violation_raw_std": violation_std,
        "adv_effective_std": float(advantage.std()),
        "adv_effective_range": float(np.ptp(advantage)),
        "safe_candidates": int(safe.sum()),
        "zero_signal_group": zero_signal,
    }
