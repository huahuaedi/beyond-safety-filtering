"""Offline diagnostic: does the correction gate respond to curvature-infeasible
raw proposals, or does it stay near zero regardless of physical feasibility?

Cross-tabulates each token's ORIGINAL (pre-correction) proposal curvature
feasibility (trajectory_dynamics.analyze_pose_trajectories) against that same
token's learned gate_probability from an already-completed eval run's output
-- no new training or data collection needed, this only reads two directories
a prior submit_eval_subset.slurm run already produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajectory_dynamics import analyze_pose_trajectories


def relative_to_absolute(increments: np.ndarray) -> np.ndarray:
    """Pure-numpy equivalent of differentiable_vehicle_model.relative_to_absolute_se2
    for a single [horizon, 3] trajectory -- no torch dependency needed here."""
    state = np.zeros(3)
    output = []
    for dx, dy, dyaw in increments:
        cosine, sine = np.cos(state[2]), np.sin(state[2])
        x = state[0] + cosine * dx - sine * dy
        y = state[1] + sine * dx + cosine * dy
        yaw = np.arctan2(np.sin(state[2] + dyaw), np.cos(state[2] + dyaw))
        state = np.array([x, y, yaw])
        output.append(state.copy())
    return np.stack(output, axis=0)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-a-dir", type=Path, required=True,
        help="output/<experiment_name>/ -- has the ORIGINAL (pre-correction) proposal per token",
    )
    parser.add_argument(
        "--unified-dir", type=Path, required=True,
        help="output/<experiment_name>_unified_correction/ -- has gate_probability per token",
    )
    return parser.parse_args()


def main():
    args = arguments()
    tokens = sorted(
        path.stem for path in args.unified_dir.glob("*.json")
        if path.stem != "unified_correction_summary"
        and not path.stem.endswith("_agent_human_traj_diff")
    )
    if not tokens:
        raise ValueError(f"no per-token JSON found under {args.unified_dir}")

    curvature_infeasible_flags, gate_probability, gate_selected = [], [], []
    for token in tokens:
        stage_a = json.loads((args.stage_a_dir / f"{token}.json").read_text())
        unified = json.loads((args.unified_dir / f"{token}.json").read_text())
        proposal = np.asarray(stage_a["action"], dtype=np.float64)
        absolute = relative_to_absolute(proposal)[None]  # [1, horizon, 3]
        dynamics = analyze_pose_trajectories(absolute, dt=0.5)
        # violation_components channel order mirrors trajectory_dynamics.py's
        # `violations` stack: [speed, accel, decel, jerk, curvature,
        # curvature_rate, friction] -- indices 4/5 are curvature-specific,
        # i.e. exactly the "尖点" (curvature-infeasible point) case.
        components = dynamics["violation_components"][0]
        curvature_infeasible_flags.append(bool(components[4] > 0.0 or components[5] > 0.0))
        gate_probability.append(float(unified["unified_correction"]["gate_probability"]))
        gate_selected.append(bool(unified["unified_correction"]["gate_selected"]))

    curvature_infeasible_flags = np.asarray(curvature_infeasible_flags)
    gate_probability = np.asarray(gate_probability)
    gate_selected = np.asarray(gate_selected)

    print(f"tokens analyzed: {len(tokens)}")
    for label, mask in (
        ("curvature-infeasible (尖点)", curvature_infeasible_flags),
        ("curvature-feasible", ~curvature_infeasible_flags),
    ):
        count = int(mask.sum())
        if count == 0:
            print(f"{label}: n=0")
            continue
        print(
            f"{label}: n={count} "
            f"mean_gate_probability={gate_probability[mask].mean():.6f} "
            f"max_gate_probability={gate_probability[mask].max():.6f} "
            f"gate_selected_rate={gate_selected[mask].mean():.4f}"
        )


if __name__ == "__main__":
    main()
