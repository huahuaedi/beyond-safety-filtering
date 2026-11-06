"""Infer bounded pseudo-controls and reconstruction metadata from trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from differentiable_vehicle_model import DifferentiableVehicleModel, VehicleModelConfig


def estimate_pseudo_controls(poses: np.ndarray, dt: float, config: VehicleModelConfig):
    """Finite-difference initialization from accumulated ``[N,H,3]`` poses."""
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (config.action_horizon, 3):
        raise ValueError(f"expected [N,{config.action_horizon},3], got {poses.shape}")
    origin = np.zeros((len(poses), 1, 3), dtype=np.float32)
    complete = np.concatenate([origin, poses], axis=1)
    distance = np.linalg.norm(np.diff(complete[..., :2], axis=1), axis=-1)
    speed = distance / dt
    heading = np.unwrap(complete[..., 2], axis=1)
    heading_delta = np.diff(heading, axis=1)
    curvature = np.divide(
        heading_delta, distance, out=np.zeros_like(heading_delta), where=distance > 1e-3
    )

    acceleration = np.zeros_like(speed)
    curvature_rate = np.zeros_like(curvature)
    acceleration[:, 1:] = np.diff(speed, axis=1) / dt
    curvature_rate[:, 1:] = np.diff(curvature, axis=1) / dt
    initial_state = np.zeros((len(poses), 5), dtype=np.float32)
    initial_state[:, 3] = speed[:, 0]
    initial_state[:, 4] = curvature[:, 0]
    controls_unbounded = np.stack([acceleration, curvature_rate], axis=-1)
    controls = controls_unbounded.copy()
    controls[..., 0] = np.clip(
        controls[..., 0], config.min_acceleration, config.max_acceleration
    )
    controls[..., 1] = np.clip(
        controls[..., 1], -config.max_abs_curvature_rate, config.max_abs_curvature_rate
    )
    initial_state[:, 3] = np.clip(initial_state[:, 3], config.min_speed, config.max_speed)
    initial_state[:, 4] = np.clip(
        initial_state[:, 4], -config.max_abs_curvature, config.max_abs_curvature
    )
    clipped = np.abs(controls - controls_unbounded) > 1e-6
    return initial_state, controls.astype(np.float32), clipped


def _controls_to_raw(controls: torch.Tensor, config: VehicleModelConfig) -> torch.Tensor:
    eps = 1e-4
    fraction = (controls[..., 0] - config.min_acceleration) / (
        config.max_acceleration - config.min_acceleration
    )
    fraction = fraction.clamp(eps, 1.0 - eps)
    zero_fraction = -config.min_acceleration / (
        config.max_acceleration - config.min_acceleration
    )
    acceleration_raw = torch.logit(fraction) - torch.logit(
        controls.new_tensor(zero_fraction)
    )
    curvature_raw = torch.atanh(
        (controls[..., 1] / config.max_abs_curvature_rate).clamp(-1.0 + eps, 1.0 - eps)
    )
    return torch.stack([acceleration_raw, curvature_raw], dim=-1)


def _raw_to_controls(raw: torch.Tensor, config: VehicleModelConfig) -> torch.Tensor:
    zero_fraction = -config.min_acceleration / (
        config.max_acceleration - config.min_acceleration
    )
    bias = torch.logit(raw.new_tensor(zero_fraction))
    acceleration = config.min_acceleration + (
        config.max_acceleration - config.min_acceleration
    ) * torch.sigmoid(raw[..., 0] + bias)
    curvature_rate = config.max_abs_curvature_rate * torch.tanh(raw[..., 1])
    return torch.stack([acceleration, curvature_rate], dim=-1)


def reconstruct_and_refine(
    poses: np.ndarray,
    initial_state: np.ndarray,
    controls: np.ndarray,
    config: VehicleModelConfig,
    *,
    steps: int,
    learning_rate: float,
    regularization: float,
):
    model = DifferentiableVehicleModel(config)
    target = torch.from_numpy(poses).float()
    initial = torch.from_numpy(initial_state).float()
    initial_controls = torch.from_numpy(controls).float()
    raw = _controls_to_raw(initial_controls, config).detach().requires_grad_(steps > 0)
    if steps > 0:
        optimizer = torch.optim.Adam([raw], lr=learning_rate)
        for _ in range(steps):
            predicted = model(initial, _raw_to_controls(raw, config))[..., :3]
            position_loss = (predicted[..., :2] - target[..., :2]).square().mean()
            heading_delta = predicted[..., 2] - target[..., 2]
            heading_loss = torch.atan2(heading_delta.sin(), heading_delta.cos()).square().mean()
            control_delta = _raw_to_controls(raw, config) - initial_controls
            smooth = torch.diff(_raw_to_controls(raw, config), dim=-2).square().mean()
            loss = position_loss + 2.0 * heading_loss + regularization * (
                control_delta.square().mean() + 0.25 * smooth
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        refined = _raw_to_controls(raw, config)
        states = model(initial, refined)
    reconstructed = states[..., :3].numpy()
    position_error = np.linalg.norm(reconstructed[..., :2] - poses[..., :2], axis=-1)
    heading_delta = reconstructed[..., 2] - poses[..., 2]
    heading_error = np.abs(np.arctan2(np.sin(heading_delta), np.cos(heading_delta)))
    return (
        refined.numpy().astype(np.float32),
        reconstructed.astype(np.float32),
        np.sqrt(np.mean(position_error**2, axis=1)).astype(np.float32),
        np.sqrt(np.mean(heading_error**2, axis=1)).astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("dynamics_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--refine-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--max-position-rmse", type=float, default=0.75)
    parser.add_argument("--max-heading-rmse", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("require 0 <= shard-id < num-shards")

    config = VehicleModelConfig()
    all_files = sorted(args.input_root.glob("*/*.npz"))
    files = all_files[args.shard_id :: args.num_shards]
    if args.max_scenes > 0:
        files = files[: args.max_scenes]
    if not files:
        raise SystemExit(f"no files assigned from {args.input_root}")

    totals = dict(candidates=0, dynamics_feasible=0, reconstruction_valid=0, valid=0, written=0, skipped=0)
    position_rmse, heading_rmse = [], []
    clipped_components = np.zeros(2, dtype=np.int64)
    for file_i, source_path in enumerate(files):
        relative = source_path.relative_to(args.input_root)
        dynamics_path = args.dynamics_root / relative
        output_path = args.output_root / relative
        if not dynamics_path.is_file():
            raise FileNotFoundError(dynamics_path)
        if output_path.exists() and not args.overwrite:
            totals["skipped"] += 1
            with np.load(output_path, allow_pickle=False) as old:
                dyn_feasible = old["dynamics_feasible"]
                recon_valid = old["reconstruction_valid"]
                pos_rmse = old["position_rmse"]
                yaw_rmse = old["heading_rmse"]
                clipped = old["analytic_control_clipped"]
        else:
            with np.load(source_path, allow_pickle=False) as data:
                token, split = str(data["scene_token"]), str(data["split"])
                poses = data["absolute_trajectories"].astype(np.float32)
            with np.load(dynamics_path, allow_pickle=False) as dynamics:
                dyn_feasible = dynamics["dynamics_feasible"].astype(bool)
            initial, analytic, clipped = estimate_pseudo_controls(poses, config.action_dt, config)
            refined, reconstructed, pos_rmse, yaw_rmse = reconstruct_and_refine(
                poses, initial, analytic, config,
                steps=args.refine_steps, learning_rate=args.learning_rate,
                regularization=args.regularization,
            )
            recon_valid = (
                (pos_rmse <= args.max_position_rmse)
                & (yaw_rmse <= args.max_heading_rmse)
                & np.isfinite(reconstructed).all(axis=(1, 2))
            )
            valid = dyn_feasible & recon_valid
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    schema_version=np.asarray(1, dtype=np.int32),
                    scene_token=np.asarray(token), split=np.asarray(split),
                    action_dt=np.asarray(config.action_dt, dtype=np.float32),
                    integration_dt=np.asarray(config.integration_dt, dtype=np.float32),
                    initial_state=initial, analytic_controls=analytic,
                    pseudo_controls=refined,
                    reconstructed_trajectory=reconstructed,
                    position_rmse=pos_rmse, heading_rmse=yaw_rmse,
                    analytic_control_clipped=clipped,
                    dynamics_feasible=dyn_feasible,
                    reconstruction_valid=recon_valid,
                    control_supervision_valid=valid,
                )
            temporary.replace(output_path)
            totals["written"] += 1
        valid = dyn_feasible & recon_valid
        totals["candidates"] += len(valid)
        totals["dynamics_feasible"] += int(dyn_feasible.sum())
        totals["reconstruction_valid"] += int(recon_valid.sum())
        totals["valid"] += int(valid.sum())
        clipped_components += clipped.sum(axis=(0, 1))
        position_rmse.append(pos_rmse)
        heading_rmse.append(yaw_rmse)
        if file_i % 100 == 0 or file_i + 1 == len(files):
            print(f"controls {file_i + 1}/{len(files)} path={output_path}", flush=True)

    positions, headings = np.concatenate(position_rmse), np.concatenate(heading_rmse)
    summary = {
        **totals,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "assigned_scenes": len(files),
        "valid_fraction": totals["valid"] / totals["candidates"],
        "position_rmse_mean": float(positions.mean()),
        "position_rmse_p50": float(np.quantile(positions, 0.5)),
        "position_rmse_p95": float(np.quantile(positions, 0.95)),
        "heading_rmse_mean": float(headings.mean()),
        "heading_rmse_p50": float(np.quantile(headings, 0.5)),
        "heading_rmse_p95": float(np.quantile(headings, 0.95)),
        "analytic_control_clipped_counts": clipped_components.tolist(),
        "refine_steps": args.refine_steps,
        "max_position_rmse": args.max_position_rmse,
        "max_heading_rmse": args.max_heading_rmse,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"summary_shard_{args.shard_id:03d}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved={summary_path}", flush=True)


if __name__ == "__main__":
    main()

