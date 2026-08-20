"""Small, framework-independent policy-optimization utilities.

The functions live outside the training entrypoint so their numerical
semantics can be unit-tested without loading the 9B policy.
"""

from __future__ import annotations

import math

import numpy as np


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
