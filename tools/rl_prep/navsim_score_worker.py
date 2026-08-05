"""
Runs inside the `navsim` conda env (not `drivevla`) — scores a batch of
candidate trajectories for one scene using the real NAVSIM PDM simulator,
replacing the nearest-anchor-lookup approximation used in Stage 1's first
pass (see docs/rl_world_model_design.md).

Called as a subprocess from rl_stage1_prototype.py (which runs in the
`drivevla` env and can't import nuplan-devkit directly — incompatible
torch/numpy pins). One subprocess call scores an entire GRPO group (default
16 candidates) so simulator/scorer/metric-cache loading cost is amortized
across the group rather than paid per candidate.

Usage:
    python navsim_score_worker.py \
        --metric-cache-path /path/to/exp/metric_cache \
        --scene-token <token> \
        --trajectories-npy /tmp/candidates.npy     # [G, 8, 3] float32, absolute
                                                     # (x, y, heading) poses in
                                                     # the ego's initial local
                                                     # frame — same format as
                                                     # relative_to_absolute_se2()
                                                     # produces in the caller
        --output-json /tmp/scores.json
"""
import argparse
import json
import lzma
import pickle
import sys
from pathlib import Path

import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.evaluate.pdm_score import pdm_score

# Matches inference/navsim/navsim/navsim/planning/script/config/pdm_scoring/
# default_scoring_parameters.yaml — keep these two in sync if that file changes.
PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)
SCORER_CONFIG = PDMScorerConfig(
    progress_weight=5.0,
    ttc_weight=5.0,
    comfortable_weight=2.0,
    driving_direction_weight=0.0,
    driving_direction_horizon=1.0,
    driving_direction_compliance_threshold=2.0,
    driving_direction_violation_threshold=6.0,
    stopped_speed_threshold=5e-03,
    progress_distance_threshold=5.0,
)
ACTION_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)  # 8 poses, matches action_frames=8

# Official PDMS weights TTC and ego_progress equally (5:5) — 阶段 2 diagnostics
# showed policy exploits this: 3x tighter KL (0.5->1.5) didn't stop held-out
# PDMS from dropping (0.9213 baseline -> ~0.85-0.89 across steps 100-400),
# and the drop is dominated by TTC collapsing while progress improves, i.e.
# the policy is trading safety margin for forward progress because the
# reward pays equally for either. A naive re-weight (TTC:progress 8:2, this
# `train_reward` field) only changes the exchange rate — a big enough
# progress gain can still buy back a safety loss. Superseded by the
# safety-gated advantage computed in rl_stage1_prototype.py (candidates that
# violate a hard safety gate never get positive advantage, no matter their
# progress) — kept here only as a comparison point in the results dict.
TRAIN_TTC_WEIGHT = 8.0
TRAIN_PROGRESS_WEIGHT = 2.0
TRAIN_COMFORT_WEIGHT = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-cache-path", required=True)
    ap.add_argument("--scene-token", required=True)
    ap.add_argument("--trajectories-npy", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    trajectories = np.load(args.trajectories_npy)  # [G, 8, 3]
    assert trajectories.ndim == 3 and trajectories.shape[1:] == (8, 3), trajectories.shape

    cache_loader = MetricCacheLoader(Path(args.metric_cache_path))
    if args.scene_token not in cache_loader.metric_cache_paths:
        print(json.dumps({"error": f"no metric cache for token {args.scene_token}"}))
        with open(args.output_json, "w") as f:
            json.dump({"error": f"no metric cache for token {args.scene_token}"}, f)
        sys.exit(1)

    cache_path = cache_loader.metric_cache_paths[args.scene_token]
    with lzma.open(cache_path, "rb") as f:
        metric_cache = pickle.load(f)

    simulator = PDMSimulator(proposal_sampling=PROPOSAL_SAMPLING)
    scorer = PDMScorer(proposal_sampling=PROPOSAL_SAMPLING, config=SCORER_CONFIG)

    results = []
    for g in range(trajectories.shape[0]):
        traj = Trajectory(poses=trajectories[g].astype(np.float32), trajectory_sampling=ACTION_SAMPLING)
        try:
            # future_sampling here must be the simulator/scorer's own
            # PROPOSAL_SAMPLING (40 poses @ 0.1s), not the trajectory's own
            # ACTION_SAMPLING (8/9 poses @ 0.5s) — see run_pdm_score.py's
            # call, which passes `future_sampling=simulator.proposal_sampling`.
            # pdm_score() resamples our trajectory onto this finer grid
            # internally; passing ACTION_SAMPLING here truncated the
            # resampled array to 9 points while the simulator's step loop
            # still iterated 40 times, causing an out-of-range slice deep in
            # BatchLQRTracker's curvature-profile computation.
            r = pdm_score(metric_cache, traj, PROPOSAL_SAMPLING, simulator, scorer)
            train_reward = (
                (TRAIN_TTC_WEIGHT * r.time_to_collision_within_bound
                 + TRAIN_PROGRESS_WEIGHT * r.ego_progress
                 + TRAIN_COMFORT_WEIGHT * r.comfort)
                / (TRAIN_TTC_WEIGHT + TRAIN_PROGRESS_WEIGHT + TRAIN_COMFORT_WEIGHT)
                * max(r.no_at_fault_collisions, 1e-6)
                * max(r.drivable_area_compliance, 1e-6)
            )
            results.append({
                "no_at_fault_collisions": float(r.no_at_fault_collisions),
                "drivable_area_compliance": float(r.drivable_area_compliance),
                "ego_progress": float(r.ego_progress),
                "time_to_collision_within_bound": float(r.time_to_collision_within_bound),
                "comfort": float(r.comfort),
                "driving_direction_compliance": float(r.driving_direction_compliance),
                "score": float(r.score),  # official PDMS — reference only, not used as reward
                "train_reward": float(train_reward),  # superseded naive re-weight — reference only
            })
        except Exception as e:  # a genuinely degenerate candidate can crash the
            # kinematic simulator (e.g. NaN pose) — score it as a hard failure
            # rather than letting one bad sample kill the whole group's scoring.
            results.append({
                "no_at_fault_collisions": 0.0, "drivable_area_compliance": 0.0,
                "ego_progress": 0.0, "time_to_collision_within_bound": 0.0,
                "comfort": 0.0, "driving_direction_compliance": 0.0,
                "score": 0.0, "train_reward": 0.0, "error": str(e),
            })

    with open(args.output_json, "w") as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    main()
