"""Project DriveVLA per-token waypoint JSONs through a control adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from build_pseudo_control_sidecars import estimate_pseudo_controls
from differentiable_vehicle_model import VehicleModelConfig, relative_to_absolute_se2
from trajectory_dynamics import analyze_pose_trajectories
from waypoint_control_adapter import WaypointControlAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens-json", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--projection-mode", default="all", choices=("all", "infeasible_only"),
        help="Project every trajectory or only trajectories failing the analytic dynamics audit.",
    )
    parser.add_argument("--max-position-rmse", type=float, default=float("inf"))
    parser.add_argument("--max-heading-rmse", type=float, default=float("inf"))
    return parser.parse_args()


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location="cpu")


def infer_initial_state(relative: np.ndarray, config: VehicleModelConfig) -> np.ndarray:
    """Infer the same non-privileged state proxy used for adapter training."""
    relative_tensor = torch.from_numpy(relative.astype(np.float32))
    absolute = relative_to_absolute_se2(relative_tensor).numpy()
    initial, _, _ = estimate_pseudo_controls(absolute, config.action_dt, config)
    return initial


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    payload = load_checkpoint(args.checkpoint)
    model = WaypointControlAdapter(
        dim=int(payload.get("dim", 128)),
        layers=int(payload.get("layers", 2)),
        heads=int(payload.get("heads", 4)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    config = model.config

    tokens = json.loads(args.tokens_json.read_text())
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"expected a non-empty token list in {args.tokens_json}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_trajectories, raw_projected_trajectories, output_trajectories = [], [], []
    position_rmse, heading_rmse = [], []
    output_position_rmse, output_heading_rmse = [], []
    attempted = accepted = rejected = 0

    with torch.inference_mode():
        for index, token in enumerate(tokens):
            input_path = args.input_dir / f"{token}.json"
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            record = json.loads(input_path.read_text())
            relative = np.asarray(record["action"], dtype=np.float32)
            if relative.shape != (config.action_horizon, 3) or not np.isfinite(relative).all():
                raise ValueError(f"{input_path}: expected finite (8,3) action, got {relative.shape}")
            relative_batch = relative[None]
            initial = infer_initial_state(relative_batch, config)
            output = model(
                torch.from_numpy(relative_batch).to(device),
                torch.from_numpy(initial).to(device),
            )
            projected_relative = output["relative_trajectory"][0].cpu().numpy()
            projected_absolute = output["absolute_trajectory"][0].cpu().numpy()
            original_absolute = relative_to_absolute_se2(
                torch.from_numpy(relative)
            ).numpy()

            original_feasible = bool(
                analyze_pose_trajectories(original_absolute[None], dt=args.dt)[
                    "dynamics_feasible"
                ][0]
            )
            projected_feasible = bool(
                analyze_pose_trajectories(projected_absolute[None], dt=args.dt)[
                    "dynamics_feasible"
                ][0]
            )

            delta_xy = np.linalg.norm(
                projected_absolute[:, :2] - original_absolute[:, :2], axis=-1
            )
            delta_yaw = projected_absolute[:, 2] - original_absolute[:, 2]
            delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
            position_rmse.append(float(np.sqrt(np.mean(delta_xy**2))))
            heading_rmse.append(float(np.sqrt(np.mean(delta_yaw**2))))
            should_attempt = args.projection_mode == "all" or not original_feasible
            should_accept = (
                should_attempt
                and projected_feasible
                and position_rmse[-1] <= args.max_position_rmse
                and heading_rmse[-1] <= args.max_heading_rmse
            )
            if should_attempt:
                attempted += 1
                if should_accept:
                    accepted += 1
                else:
                    rejected += 1
            selected_relative = projected_relative if should_accept else relative
            selected_absolute = projected_absolute if should_accept else original_absolute
            output_position_rmse.append(position_rmse[-1] if should_accept else 0.0)
            output_heading_rmse.append(heading_rmse[-1] if should_accept else 0.0)
            original_trajectories.append(original_absolute)
            raw_projected_trajectories.append(projected_absolute)
            output_trajectories.append(selected_absolute)

            transformed = dict(record)
            transformed["action"] = selected_relative.astype(float).tolist()
            transformed["control_adapter"] = {
                "checkpoint": str(args.checkpoint),
                "projection_mode": args.projection_mode,
                "original_feasible": original_feasible,
                "projected_feasible": projected_feasible,
                "attempted": should_attempt,
                "accepted": should_accept,
                "initial_state_proxy": initial[0].astype(float).tolist(),
                "controls": output["controls"][0].cpu().numpy().astype(float).tolist(),
                "position_rmse_to_original": position_rmse[-1],
                "heading_rmse_to_original": heading_rmse[-1],
            }
            (args.output_dir / f"{token}.json").write_text(
                json.dumps(transformed, indent=2) + "\n"
            )
            if index % 50 == 0 or index + 1 == len(tokens):
                print(f"projected {index + 1}/{len(tokens)} tokens", flush=True)

    original = np.stack(original_trajectories)
    raw_projected = np.stack(raw_projected_trajectories)
    output = np.stack(output_trajectories)
    original_dynamics = analyze_pose_trajectories(original, dt=args.dt)
    raw_projected_dynamics = analyze_pose_trajectories(raw_projected, dt=args.dt)
    output_dynamics = analyze_pose_trajectories(output, dt=args.dt)
    original_feasible = original_dynamics["dynamics_feasible"]
    raw_projected_feasible = raw_projected_dynamics["dynamics_feasible"]
    output_feasible = output_dynamics["dynamics_feasible"]
    invalid_mask = ~original_feasible
    invalid_position = np.asarray(position_rmse)[invalid_mask]
    invalid_heading = np.asarray(heading_rmse)[invalid_mask]
    summary = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "tokens": len(tokens),
        "projection_mode": args.projection_mode,
        "max_position_rmse": args.max_position_rmse,
        "max_heading_rmse": args.max_heading_rmse,
        "projection_attempted": attempted,
        "projection_accepted": accepted,
        "projection_rejected": rejected,
        "original_infeasible": int((~original_feasible).sum()),
        "raw_projected_infeasible": int((~raw_projected_feasible).sum()),
        "output_infeasible": int((~output_feasible).sum()),
        "infeasible_repaired": int((~original_feasible & output_feasible).sum()),
        "feasible_regressed": int((original_feasible & ~output_feasible).sum()),
        "position_rmse_to_original": {
            "mean": float(np.mean(position_rmse)),
            "p50": float(np.quantile(position_rmse, 0.50)),
            "p95": float(np.quantile(position_rmse, 0.95)),
        },
        "heading_rmse_to_original": {
            "mean": float(np.mean(heading_rmse)),
            "p50": float(np.quantile(heading_rmse, 0.50)),
            "p95": float(np.quantile(heading_rmse, 0.95)),
        },
        "actual_output_position_rmse_to_original": {
            "mean": float(np.mean(output_position_rmse)),
            "p50": float(np.quantile(output_position_rmse, 0.50)),
            "p95": float(np.quantile(output_position_rmse, 0.95)),
        },
        "actual_output_heading_rmse_to_original": {
            "mean": float(np.mean(output_heading_rmse)),
            "p50": float(np.quantile(output_heading_rmse, 0.50)),
            "p95": float(np.quantile(output_heading_rmse, 0.95)),
        },
        "originally_infeasible_projection_error": {
            "count": int(invalid_mask.sum()),
            "position_rmse_mean": float(invalid_position.mean()) if len(invalid_position) else None,
            "position_rmse_p95": float(np.quantile(invalid_position, 0.95)) if len(invalid_position) else None,
            "heading_rmse_mean": float(invalid_heading.mean()) if len(invalid_heading) else None,
            "heading_rmse_p95": float(np.quantile(invalid_heading, 0.95)) if len(invalid_heading) else None,
        },
    }
    (args.output_dir / "control_adapter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
