"""Pre-2026-09-23 reference-anchored advantage, kept for A/B comparison only.

Source: tools/rl_prep/rl_stage1_prototype.py at commit c648a3f (the 2026-09-07
HPC snapshot). Training logs of the original ``direct_rl_seed20-24`` runs show
``adv_mean == 0`` in every step, which matches this group-centred form and not
the current hard-sign gate (``0.5*tanh`` for feasible candidates, no centring).

Differences from ``policy_optimization.compute_reference_anchored_advantage``:
  * feasible candidates keep unit-scale z-scores instead of ``0.5*tanh(z)``;
  * the whole group is re-centred to zero mean, so a feasible candidate can
    receive a positive advantage only relative to the group, and an infeasible
    one can end up above zero when most of the group is infeasible;
  * a single feasible candidate gets +1 before centring.
The Aug-29 direct_rl code itself was not recovered; this is the closest
preserved version, so treat the A/B as "legacy-style", not "bit-identical".
"""
from __future__ import annotations

import numpy as np

from adaptive_reward import validate_constraint_weights
from policy_optimization import shrinkage_centered_advantage


def legacy_reference_anchored_advantage(
    sub_scores: dict, reference_scores: dict, constraint_weights=None, *, noise_floor=0.0
):
    nc = sub_scores["no_at_fault_collisions"]
    dac = sub_scores["drivable_area_compliance"]
    ttc = sub_scores["time_to_collision_within_bound"]
    progress = sub_scores["ego_progress"] + 0.25 * sub_scores["driving_direction_compliance"]
    ref_nc = float(reference_scores["no_at_fault_collisions"])
    ref_dac = float(reference_scores["drivable_area_compliance"])
    ref_ttc = float(reference_scores["time_to_collision_within_bound"])
    ref_utility = float(reference_scores["ego_progress"]) + 0.25 * float(
        reference_scores["driving_direction_compliance"]
    )

    weights = validate_constraint_weights(
        constraint_weights or {"nc": 1.0, "dac": 1.0, "ttc": 1.0}
    )
    nc_regret = np.maximum(0.0, ref_nc - nc)
    dac_regret = np.maximum(0.0, ref_dac - dac)
    ttc_regret = np.maximum(0.0, ref_ttc - ttc)
    raw_violation = nc_regret + dac_regret + ttc_regret
    feasible = raw_violation <= 1e-8
    violation = (
        weights["nc"] * nc_regret + weights["dac"] * dac_regret + weights["ttc"] * ttc_regret
    )
    adv = np.zeros(len(nc), dtype=np.float32)
    n_feasible = int(feasible.sum())

    if n_feasible >= 2:
        improvement = progress[feasible] - ref_utility
        adv[feasible] = shrinkage_centered_advantage(improvement, noise_floor=noise_floor)
        if (~feasible).any():
            bad = violation[~feasible]
            adv[~feasible] = -1.0 - bad / (bad.max() + 1e-6)
        adv -= adv.mean()
    elif n_feasible == 1:
        adv[feasible] = 1.0
        if (~feasible).any():
            bad = violation[~feasible]
            adv[~feasible] = -1.0 - bad / (bad.max() + 1e-6)
        adv -= adv.mean()
    else:
        adv = -shrinkage_centered_advantage(violation, noise_floor=noise_floor)

    return adv.astype(np.float32), feasible, violation.astype(np.float32)
