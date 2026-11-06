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
        --trajectories-npy /tmp/candidates.npy \   # [G, 8, 3] float32, absolute
                                                     # (x, y, heading) poses in
                                                     # the ego's initial local
                                                     # frame — same format as
                                                     # relative_to_absolute_se2()
                                                     # produces in the caller
        --output-json /tmp/scores.json \
        [--trace-npz /tmp/trace.npz]           # optional: keep desired vs
                                                # executed states, tracking
                                                # error and drivable margins
                                                # (see execution_trace.py)

Without --trace-npz the JSON output is unchanged. With it, each candidate is
scored by pdm_score_with_trace(), a copy of navsim's pdm_score() that also
returns the simulator/scorer internals pdm_score() discards; the official
sub-scores are computed by the same calls in the same order.
"""
import argparse
import json
import lzma
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex

import execution_trace as et

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


class TraceScoreMismatch(RuntimeError):
    pass


def pdm_score_with_trace(metric_cache, model_trajectory, simulator, scorer, open_loop_scorer):
    """navsim.evaluate.pdm_score.pdm_score() plus the states it discards.

    The official result is read from `scorer` exactly as pdm_score() does.
    The desired states are then scored open-loop (no tracker) by a separate
    scorer instance, with finite-difference velocities so the scorer's
    collision classification sees a moving ego. The PDM-Closed reference
    stays at index 0 in both calls, as in pdm_score().
    """
    initial_ego_state = metric_cache.ego_state
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)
    pdm_states, pred_states = (
        get_trajectory_as_array(metric_cache.trajectory, PROPOSAL_SAMPLING, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, PROPOSAL_SAMPLING, initial_ego_state.time_point),
    )
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
    scene_args = (
        metric_cache.observation, metric_cache.centerline,
        metric_cache.route_lane_ids, metric_cache.drivable_area_map,
    )
    scores = scorer.score_proposals(simulated_states, *scene_args)
    pred_idx = 1
    result = PDMResults(
        scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx],
        scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx],
        scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx],
        scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx],
        scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx],
        scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx],
        scores[pred_idx],
    )
    trace = {
        "desired_states": pred_states,
        "executed_states": simulated_states[pred_idx],
        "pdm_reference_executed_states": simulated_states[0],
        "executed_ego_areas": scorer._ego_areas[pred_idx].copy(),
        "collision_time_idx": float(scorer._collision_time_idcs[pred_idx]),
        "ttc_time_idx": float(scorer._ttc_time_idcs[pred_idx]),
        "progress_raw": float(scorer._progress_raw[pred_idx]),
    }
    # Diagnostic only: a failure here must not turn an officially scored
    # candidate into a hard-failure row, so it only blanks the desired fields.
    try:
        desired_with_dynamics = et.fill_finite_difference_dynamics(pred_states, PROPOSAL_SAMPLING.interval_length)
        open_loop_scorer.score_proposals(
            np.stack([simulated_states[0], desired_with_dynamics]), *scene_args
        )
        trace["desired_ego_areas"] = open_loop_scorer._ego_areas[pred_idx].copy()
        trace["desired__no_at_fault_collisions"] = float(open_loop_scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx])
        trace["desired__drivable_area_compliance"] = float(open_loop_scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx])
        trace["desired__time_to_collision_within_bound"] = float(open_loop_scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx])
    except Exception as e:
        trace["desired_ego_areas"] = np.zeros_like(trace["executed_ego_areas"])
        for key in ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound"):
            trace[f"desired__{key}"] = float("nan")
        trace["open_loop_error"] = str(e)
    return result, trace


def drivable_polygons(drivable_area_map):
    # Same layers the scorer uses for its NON_DRIVABLE_AREA flag.
    idcs = drivable_area_map.get_indices_of_map_type([
        SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA,
    ])
    return [drivable_area_map._geometries[i] for i in idcs]


def write_trace(path, scene_token, meta_json, trajectories, metric_cache, rows, traces, errors):
    vehicle = metric_cache.ego_state.car_footprint.vehicle_parameters
    contains_xy, boundary_distance = et.shapely_drivable_callables(
        drivable_polygons(metric_cache.drivable_area_map)
    )
    payload = et.build_trace_payload(
        scene_token, meta_json, trajectories,
        (vehicle.half_length, vehicle.half_width, vehicle.rear_axle_to_center, vehicle.wheel_base),
        rows, traces, errors, contains_xy, boundary_distance,
        num_states=PROPOSAL_SAMPLING.num_poses + 1,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp_path, path)


def result_row(r):
    train_reward = (
        (TRAIN_TTC_WEIGHT * r.time_to_collision_within_bound
         + TRAIN_PROGRESS_WEIGHT * r.ego_progress
         + TRAIN_COMFORT_WEIGHT * r.comfort)
        / (TRAIN_TTC_WEIGHT + TRAIN_PROGRESS_WEIGHT + TRAIN_COMFORT_WEIGHT)
        * max(r.no_at_fault_collisions, 1e-6)
        * max(r.drivable_area_compliance, 1e-6)
    )
    return {
        "no_at_fault_collisions": float(r.no_at_fault_collisions),
        "drivable_area_compliance": float(r.drivable_area_compliance),
        "ego_progress": float(r.ego_progress),
        "time_to_collision_within_bound": float(r.time_to_collision_within_bound),
        "comfort": float(r.comfort),
        "driving_direction_compliance": float(r.driving_direction_compliance),
        "score": float(r.score),  # official PDMS — reference only, not used as reward
        "train_reward": float(train_reward),  # superseded naive re-weight — reference only
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-cache-path", required=True)
    ap.add_argument("--scene-token", required=True)
    ap.add_argument("--trajectories-npy", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--trace-npz", default="")
    ap.add_argument("--trace-meta-json", default="{}")
    # Smoke-test switch: also run the stock pdm_score() per candidate and
    # abort if the traced path's official sub-scores differ at all.
    ap.add_argument("--verify-trace-scores", action="store_true")
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

    open_loop_scorer = PDMScorer(proposal_sampling=PROPOSAL_SAMPLING, config=SCORER_CONFIG) if args.trace_npz else None

    results, traces, errors = [], [], []
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
            if args.trace_npz:
                r, trace = pdm_score_with_trace(metric_cache, traj, simulator, scorer, open_loop_scorer)
                if args.verify_trace_scores:
                    stock = pdm_score(metric_cache, traj, PROPOSAL_SAMPLING, simulator, scorer)
                    if result_row(stock) != result_row(r):
                        raise TraceScoreMismatch(f"traced scores differ from pdm_score(): {result_row(r)} vs {result_row(stock)}")
            else:
                r, trace = pdm_score(metric_cache, traj, PROPOSAL_SAMPLING, simulator, scorer), None
            results.append(result_row(r))
            traces.append(trace)
            errors.append("")
        except TraceScoreMismatch:
            raise
        except Exception as e:  # a genuinely degenerate candidate can crash the
            # kinematic simulator (e.g. NaN pose) — score it as a hard failure
            # rather than letting one bad sample kill the whole group's scoring.
            results.append({
                "no_at_fault_collisions": 0.0, "drivable_area_compliance": 0.0,
                "ego_progress": 0.0, "time_to_collision_within_bound": 0.0,
                "comfort": 0.0, "driving_direction_compliance": 0.0,
                "score": 0.0, "train_reward": 0.0, "error": str(e),
            })
            traces.append(None)
            errors.append(str(e))

    if args.trace_npz:
        write_trace(args.trace_npz, args.scene_token, args.trace_meta_json, trajectories,
                    metric_cache, results, traces, errors)

    with open(args.output_json, "w") as f:
        json.dump({"results": results}, f)


if __name__ == "__main__":
    main()
