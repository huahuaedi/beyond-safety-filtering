"""Quality/advantage metadata for context-conditioned DriveVLA training.

The metadata is deliberately vector-valued: collision, drivable-area and TTC
regressions remain distinguishable instead of being collapsed into one reward.
All labels are reference-relative and computed independently within a scene.
"""

from __future__ import annotations

import numpy as np


QUALITY_NAMES = ("critical", "recovery", "safe", "high_quality")
RISK_NAMES = ("nc", "dac", "ttc")
CONDITION_DIM = 11


def _as_candidate_array(values, size, name):
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {(size,)}, got {array.shape}")
    return array


def _rank_bins(values, bins=5):
    """Stable, approximately balanced within-scene rank bins in [0, bins)."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return np.minimum((ranks * bins) // max(len(values), 1), bins - 1).astype(np.int8)


def build_quality_metadata(scores, reference_scores, constraint_weights=None):
    """Return candidate-level quality labels and a compact condition vector."""
    size = len(scores["ego_progress"])
    if size == 0:
        raise ValueError("a scene must contain at least one candidate")
    nc = _as_candidate_array(scores["no_at_fault_collisions"], size, "nc")
    dac = _as_candidate_array(scores["drivable_area_compliance"], size, "dac")
    ttc = _as_candidate_array(scores["time_to_collision_within_bound"], size, "ttc")
    progress = _as_candidate_array(scores["ego_progress"], size, "progress")
    direction = _as_candidate_array(
        scores["driving_direction_compliance"], size, "direction"
    )
    weights = {"nc": 1.0, "dac": 1.0, "ttc": 1.0}
    if constraint_weights is not None:
        weights.update({key: float(constraint_weights[key]) for key in RISK_NAMES})
    if any(not np.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError(f"constraint weights must be finite and positive: {weights}")

    ref_nc = float(reference_scores["no_at_fault_collisions"])
    ref_dac = float(reference_scores["drivable_area_compliance"])
    ref_ttc = float(reference_scores["time_to_collision_within_bound"])
    ref_utility = float(reference_scores["ego_progress"]) + 0.25 * float(
        reference_scores["driving_direction_compliance"]
    )
    utility = progress + 0.25 * direction
    improvement = utility - ref_utility

    component_regret = np.stack(
        [
            np.maximum(0.0, ref_nc - nc),
            np.maximum(0.0, ref_dac - dac),
            np.maximum(0.0, ref_ttc - ttc),
        ],
        axis=1,
    ).astype(np.float32)
    risk_mask = component_regret > 1e-8
    raw_violation = component_regret.sum(axis=1)
    weighted_violation = (
        component_regret
        * np.asarray([weights["nc"], weights["dac"], weights["ttc"]], dtype=np.float32)
    ).sum(axis=1)
    feasible = raw_violation <= 1e-8

    advantage = np.zeros(size, dtype=np.float32)
    if feasible.sum() >= 2:
        safe_improvement = improvement[feasible]
        advantage[feasible] = (
            safe_improvement - safe_improvement.mean()
        ) / (safe_improvement.std() + 1e-6)
        if (~feasible).any():
            bad = weighted_violation[~feasible]
            advantage[~feasible] = -1.0 - bad / (bad.max() + 1e-6)
        advantage -= advantage.mean()
    elif feasible.sum() == 1:
        advantage[feasible] = 1.0
        if (~feasible).any():
            bad = weighted_violation[~feasible]
            advantage[~feasible] = -1.0 - bad / (bad.max() + 1e-6)
        advantage -= advantage.mean()
    else:
        advantage = -(
            weighted_violation - weighted_violation.mean()
        ) / (weighted_violation.std() + 1e-6)

    quality_tier = np.zeros(size, dtype=np.int8)
    if feasible.any():
        high_threshold = float(np.quantile(improvement[feasible], 0.75))
        quality_tier[feasible] = 2
        quality_tier[feasible & (improvement >= high_threshold)] = 3
    if (~feasible).any():
        recovery_threshold = float(np.median(weighted_violation[~feasible]))
        quality_tier[(~feasible) & (weighted_violation <= recovery_threshold)] = 1

    advantage_bin = _rank_bins(advantage, bins=5)
    condition = np.zeros((size, CONDITION_DIM), dtype=np.float32)
    condition[np.arange(size), quality_tier] = 1.0
    condition[:, 4:7] = risk_mask.astype(np.float32)
    condition[:, 7] = feasible.astype(np.float32)
    condition[:, 8] = np.clip(advantage / 3.0, -1.0, 1.0)
    condition[:, 9] = np.clip(improvement, -1.0, 1.0)
    condition[:, 10] = np.clip(weighted_violation / 3.0, 0.0, 1.0)

    return {
        "quality_tier": quality_tier,
        "advantage": advantage.astype(np.float32),
        "advantage_bin": advantage_bin,
        "risk_mask": risk_mask.astype(np.bool_),
        "component_regret": component_regret,
        "weighted_violation": weighted_violation.astype(np.float32),
        "utility_improvement": improvement.astype(np.float32),
        "condition_vector": condition,
    }


def desired_high_quality_condition(size=1):
    """Inference-time request: high quality, feasible and no known risk."""
    condition = np.zeros((int(size), CONDITION_DIM), dtype=np.float32)
    condition[:, 3] = 1.0
    condition[:, 7] = 1.0
    condition[:, 8] = 1.0
    condition[:, 9] = 1.0
    return condition
