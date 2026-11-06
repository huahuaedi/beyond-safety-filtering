"""Bounded scene-level safety weights derived from WM risk predictions."""

import math

import numpy as np


CONSTRAINT_KEYS = ("nc", "dac", "ttc")
# Risk-ranker label order: unsafe, TTC failure, DAC failure, NC failure.
_PROBABILITY_COLUMNS = {"nc": 3, "dac": 2, "ttc": 1}


def validate_constraint_weights(weights, *, minimum=0.0, maximum=float("inf")):
    """Validate and canonicalize an NC/DAC/TTC weight mapping."""
    canonical = {key: float(weights[key]) for key in CONSTRAINT_KEYS}
    for key, value in canonical.items():
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"constraint weight {key}={value!r} must be finite and in "
                f"[{minimum}, {maximum}]"
            )
    return canonical


def scene_constraint_weights(
    candidate_probabilities,
    *,
    topk=8,
    strength=0.5,
    minimum=0.5,
    maximum=2.0,
    risk_baselines=None,
):
    """Turn candidate risk probabilities into shared, bounded scene weights.

    A top-k mean summarizes tail risk per constraint. Relative salience is
    shrunk toward one so noisy heads cannot dominate. Sharing one weight set
    across the scene preserves valid GRPO within-group comparisons.
    """
    probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError(
            "candidate probabilities must have shape [num_candidates, 4], "
            f"got {probabilities.shape}"
        )
    if probabilities.shape[0] == 0 or not np.isfinite(probabilities).all():
        raise ValueError("candidate probabilities must be non-empty and finite")
    topk = int(topk)
    if not 1 <= topk <= probabilities.shape[0]:
        raise ValueError(f"topk must be in [1, {probabilities.shape[0]}]")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if not 0.0 < float(minimum) <= 1.0 <= float(maximum):
        raise ValueError("bounds must satisfy 0 < minimum <= 1 <= maximum")

    tail_risk = {}
    for key, column in _PROBABILITY_COLUMNS.items():
        values = probabilities[:, column]
        tail_risk[key] = float(np.partition(values, -topk)[-topk:].mean())
    baselines = validate_constraint_weights(
        risk_baselines or {key: 1.0 for key in CONSTRAINT_KEYS}, minimum=1e-8
    )
    # Compare each head to its own typical tail-risk scale. Risk-ranker
    # probabilities are not calibrated across labels because each label has a
    # different positive-class weight; raw cross-label magnitudes would, for
    # example, systematically suppress rare collision risk.
    relative_risk = {
        key: tail_risk[key] / baselines[key] for key in CONSTRAINT_KEYS
    }
    mean_risk = max(sum(relative_risk.values()) / len(relative_risk), 1e-8)
    weights = {
        key: float(np.clip(1.0 + strength * (risk / mean_risk - 1.0), minimum, maximum))
        for key, risk in relative_risk.items()
    }
    return validate_constraint_weights(weights, minimum=minimum, maximum=maximum), tail_risk


def estimate_tail_risk_baselines(scene_candidate_probabilities, *, topk=8):
    """Estimate each head's typical scene tail from OOF predictions only."""
    probabilities = np.asarray(scene_candidate_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[2] != 4:
        raise ValueError(
            "scene candidate probabilities must have shape [scenes, candidates, 4], "
            f"got {probabilities.shape}"
        )
    tails = {key: [] for key in CONSTRAINT_KEYS}
    for scene in probabilities:
        _, scene_tail = scene_constraint_weights(scene, topk=topk)
        for key in CONSTRAINT_KEYS:
            tails[key].append(scene_tail[key])
    baselines = {key: float(np.mean(values)) for key, values in tails.items()}
    return validate_constraint_weights(baselines, minimum=1e-8)
