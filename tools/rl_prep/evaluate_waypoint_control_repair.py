"""Evaluate whether a trained waypoint adapter preserves or repairs dynamics.

Unlike the training-time evaluation, this script scores every candidate in the
test split, including candidates excluded from pseudo-control supervision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trajectory_dynamics import analyze_pose_trajectories
from waypoint_control_adapter import WaypointControlAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.5)
    return parser.parse_args()


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This is a trusted checkpoint produced by our own training script.  Old
    # PyTorch releases reject weights_only while newer releases default it to
    # True, so support both APIs explicitly.
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        payload = torch.load(args.checkpoint, map_location="cpu")
    model = WaypointControlAdapter(
        dim=int(payload.get("dim", 128)),
        layers=int(payload.get("layers", 2)),
        heads=int(payload.get("heads", 4)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    paths = sorted((args.data_root / args.split).glob("*.npz"))
    if args.max_scenes > 0:
        paths = paths[: args.max_scenes]
    if not paths:
        raise FileNotFoundError(f"no {args.split} files under {args.data_root}")

    original_feasible_parts, predicted_feasible_parts = [], []
    position_rmse_parts, heading_rmse_parts = [], []
    predicted_component_parts = []
    candidates = 0
    with torch.inference_mode():
        for scene_i, path in enumerate(paths):
            sidecar_path = args.control_root / args.split / path.name
            if not sidecar_path.is_file():
                raise FileNotFoundError(sidecar_path)
            with np.load(path, allow_pickle=False) as data:
                relative = data["relative_trajectories"].astype(np.float32)
                absolute = data["absolute_trajectories"].astype(np.float32)
            with np.load(sidecar_path, allow_pickle=False) as data:
                initial = data["initial_state"].astype(np.float32)
                original_feasible = data["dynamics_feasible"].astype(bool)

            for start in range(0, len(relative), args.batch_size):
                stop = min(start + args.batch_size, len(relative))
                relative_batch = torch.from_numpy(relative[start:stop]).to(device)
                initial_batch = torch.from_numpy(initial[start:stop]).to(device)
                predicted = model(relative_batch, initial_batch)["absolute_trajectory"].cpu().numpy()
                target = absolute[start:stop]
                position_error = np.linalg.norm(predicted[..., :2] - target[..., :2], axis=-1)
                heading_delta = np.arctan2(
                    np.sin(predicted[..., 2] - target[..., 2]),
                    np.cos(predicted[..., 2] - target[..., 2]),
                )
                dynamics = analyze_pose_trajectories(predicted, dt=args.dt)
                original_feasible_parts.append(original_feasible[start:stop])
                predicted_feasible_parts.append(dynamics["dynamics_feasible"])
                predicted_component_parts.append(dynamics["violation_components"] > 0)
                position_rmse_parts.append(np.sqrt(np.mean(position_error**2, axis=1)))
                heading_rmse_parts.append(np.sqrt(np.mean(heading_delta**2, axis=1)))
                candidates += stop - start
            if scene_i % 100 == 0 or scene_i + 1 == len(paths):
                print(f"evaluated {scene_i + 1}/{len(paths)} scenes", flush=True)

    original = np.concatenate(original_feasible_parts)
    predicted = np.concatenate(predicted_feasible_parts)
    components = np.concatenate(predicted_component_parts)
    position = np.concatenate(position_rmse_parts)
    heading = np.concatenate(heading_rmse_parts)
    originally_feasible = int(original.sum())
    originally_infeasible = int((~original).sum())
    preserved = int((original & predicted).sum())
    repaired = int((~original & predicted).sum())
    regressed = int((original & ~predicted).sum())
    still_infeasible = int((~original & ~predicted).sum())
    component_names = [
        "speed", "acceleration", "deceleration", "jerk", "curvature",
        "curvature_rate", "lateral_acceleration",
    ]
    results = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "device": str(device),
        "scenes": len(paths),
        "candidates": candidates,
        "originally_feasible": originally_feasible,
        "originally_infeasible": originally_infeasible,
        "predicted_feasible": int(predicted.sum()),
        "predicted_infeasible": int((~predicted).sum()),
        "feasible_preservation_rate": safe_rate(preserved, originally_feasible),
        "infeasible_repair_rate": safe_rate(repaired, originally_infeasible),
        "feasible_regression_rate": safe_rate(regressed, originally_feasible),
        "unrepaired_infeasible_rate": safe_rate(still_infeasible, originally_infeasible),
        "transition_counts": {
            "feasible_to_feasible": preserved,
            "feasible_to_infeasible": regressed,
            "infeasible_to_feasible": repaired,
            "infeasible_to_infeasible": still_infeasible,
        },
        "position_rmse": {
            "mean": float(position.mean()),
            "p50": float(np.quantile(position, 0.50)),
            "p95": float(np.quantile(position, 0.95)),
        },
        "heading_rmse": {
            "mean": float(heading.mean()),
            "p50": float(np.quantile(heading, 0.50)),
            "p95": float(np.quantile(heading, 0.95)),
        },
        "predicted_component_failure_counts": {
            name: int(components[:, index].sum())
            for index, name in enumerate(component_names)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
