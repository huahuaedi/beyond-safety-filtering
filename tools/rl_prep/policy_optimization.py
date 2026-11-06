"""Small, framework-independent policy-optimization utilities.

The functions live outside the training entrypoint so their numerical
semantics can be unit-tested without loading the 9B policy.
"""

from __future__ import annotations

import math

import numpy as np

from adaptive_reward import validate_constraint_weights


def shrinkage_centered_advantage(
    values: np.ndarray,
    *,
    noise_floor: float = 0.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Center values and normalize with an explicit observation-noise floor.

    With ``noise_floor=0`` this matches the legacy group z-score (apart from
    immaterial floating-point rearrangement).  A positive floor prevents tiny
    raw score differences from being amplified to unit-variance advantages.
    """
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"values must be a non-empty vector, got {x.shape}")
    if noise_floor < 0 or epsilon <= 0:
        raise ValueError("noise_floor must be >= 0 and epsilon must be > 0")
    centered = x - x.mean()
    std = float(x.std())
    if noise_floor == 0:
        denominator = std + epsilon
    else:
        denominator = math.sqrt(std * std + noise_floor * noise_floor + epsilon * epsilon)
    return (centered / denominator).astype(np.float32)


def update_adaptive_kl_coefficient(
    coefficient: float,
    observed_kl: float,
    *,
    target_kl: float,
    adaptation_rate: float,
    minimum: float,
    maximum: float,
) -> float:
    """Multiplicative proportional controller for reference-KL pressure."""
    if coefficient <= 0 or target_kl <= 0 or adaptation_rate < 0:
        raise ValueError("coefficient/target_kl must be > 0 and adaptation_rate >= 0")
    if not 0 < minimum <= maximum:
        raise ValueError("KL coefficient bounds must satisfy 0 < min <= max")
    if not math.isfinite(observed_kl) or observed_kl < 0:
        raise ValueError("observed_kl must be finite and non-negative")
    # Limit a single observation's influence; the controller should adapt,
    # not jump orders of magnitude because of one noisy group.
    error = max(-1.0, min(1.0, observed_kl / target_kl - 1.0))
    updated = coefficient * math.exp(adaptation_rate * error)
    return float(min(maximum, max(minimum, updated)))



def compute_reference_anchored_advantage(
    sub_scores: dict, reference_scores: dict, constraint_weights=None, *, noise_floor=0.0
):
    """Hard-sign, reference-safety-gated advantage, not a safety guarantee.

    Feasible utility ranks lie in [-0.5, 0.5] (singleton: +0.5).
    Regressing candidates lie in [-2, -1], including all-regressing groups:
    recovery suppresses every sampled regression, never rewards the least bad.
    Do not re-center the complete group; that can reverse the hard sign gate.
    Utility is group-relative, not an absolute improvement over reference.
    """
    nc = sub_scores["no_at_fault_collisions"]
    dac = sub_scores["drivable_area_compliance"]
    ttc = sub_scores["time_to_collision_within_bound"]
    progress = (
        sub_scores["ego_progress"]
        + 0.25 * sub_scores["driving_direction_compliance"]
    )
    ref_nc = float(reference_scores["no_at_fault_collisions"])
    ref_dac = float(reference_scores["drivable_area_compliance"])
    ref_ttc = float(reference_scores["time_to_collision_within_bound"])

    weights = validate_constraint_weights(
        constraint_weights or {"nc": 1.0, "dac": 1.0, "ttc": 1.0}
    )
    nc_regret = np.maximum(0.0, ref_nc - nc)
    dac_regret = np.maximum(0.0, ref_dac - dac)
    ttc_regret = np.maximum(0.0, ref_ttc - ttc)
    # The hard feasibility gate remains independent of learned weights.
    raw_violation = nc_regret + dac_regret + ttc_regret
    feasible = raw_violation <= 1e-8
    violation = (
        weights["nc"] * nc_regret
        + weights["dac"] * dac_regret
        + weights["ttc"] * ttc_regret
    )
    adv = np.zeros(len(nc), dtype=np.float32)
    n_feasible = int(feasible.sum())

    if n_feasible >= 2:
        relative = shrinkage_centered_advantage(
            progress[feasible], noise_floor=noise_floor
        )
        adv[feasible] = 0.5 * np.tanh(relative)
    elif n_feasible == 1:
        adv[feasible] = 0.5
    if (~feasible).any():
        bad = violation[~feasible]
        adv[~feasible] = -1.0 - bad / (bad.max() + 1e-6)

    return adv.astype(np.float32), feasible, violation.astype(np.float32)
