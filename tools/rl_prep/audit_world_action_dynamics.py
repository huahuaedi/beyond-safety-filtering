"""Create per-scene dynamics sidecars for a world-action dataset shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajectory_dynamics import DynamicsLimits, analyze_pose_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-speed", type=float, default=20.0)
    parser.add_argument("--max-acceleration", type=float, default=4.0)
    parser.add_argument("--max-deceleration", type=float, default=6.0)
    parser.add_argument("--max-abs-jerk", type=float, default=8.0)
    parser.add_argument("--max-abs-curvature", type=float, default=0.25)
    parser.add_argument("--max-abs-curvature-rate", type=float, default=0.20)
    parser.add_argument("--max-lateral-acceleration", type=float, default=4.0)
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("require 0 <= shard-id < num-shards")

    limits = DynamicsLimits(
        max_speed=args.max_speed,
        max_acceleration=args.max_acceleration,
        max_deceleration=args.max_deceleration,
        max_abs_jerk=args.max_abs_jerk,
        max_abs_curvature=args.max_abs_curvature,
        max_abs_curvature_rate=args.max_abs_curvature_rate,
        max_lateral_acceleration=args.max_lateral_acceleration,
    )
    all_files = sorted(args.input_root.glob("*/*.npz"))
    files = all_files[args.shard_id :: args.num_shards]
    if not files:
        raise SystemExit(f"no files assigned from {args.input_root}")

    candidates = infeasible = written = skipped = 0
    component_counts = np.zeros(7, dtype=np.int64)
    for file_i, source_path in enumerate(files):
        relative = source_path.relative_to(args.input_root)
        output_path = args.output_root / relative
        if output_path.exists() and not args.overwrite:
            skipped += 1
            with np.load(output_path, allow_pickle=False) as old:
                feasible = old["dynamics_feasible"]
                violations = old["violation_components"]
        else:
            with np.load(source_path, allow_pickle=False) as data:
                token = str(data["scene_token"])
                split = str(data["split"])
                # DriveVLA's relative_trajectories are per-step SE(2)
                # increments, not future poses. Finite differences must use
                # the already accumulated poses that NAVSIM receives.
                trajectories = data["absolute_trajectories"].astype(np.float32)
            result = analyze_pose_trajectories(trajectories, args.dt, limits)
            feasible = result["dynamics_feasible"]
            violations = result["violation_components"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    schema_version=np.asarray(1, dtype=np.int32),
                    scene_token=np.asarray(token),
                    split=np.asarray(split),
                    dt=np.asarray(args.dt, dtype=np.float32),
                    limit_names=np.asarray([
                        "speed", "acceleration", "deceleration", "jerk",
                        "curvature", "curvature_rate", "lateral_acceleration",
                    ]),
                    limits=np.asarray([
                        limits.max_speed, limits.max_acceleration, limits.max_deceleration,
                        limits.max_abs_jerk, limits.max_abs_curvature,
                        limits.max_abs_curvature_rate, limits.max_lateral_acceleration,
                    ], dtype=np.float32),
                    **result,
                )
            temporary.replace(output_path)
            written += 1
        candidates += len(feasible)
        infeasible += int((~feasible).sum())
        component_counts += (violations > 0).sum(axis=0)
        if file_i % 250 == 0 or file_i + 1 == len(files):
            print(f"dynamics {file_i + 1}/{len(files)} path={output_path}", flush=True)

    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "assigned_scenes": len(files),
        "written": written,
        "skipped": skipped,
        "candidates": candidates,
        "infeasible": infeasible,
        "infeasible_fraction": infeasible / candidates,
        "component_failure_counts": component_counts.tolist(),
        "dt": args.dt,
        "limits": limits.to_dict(),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"summary_shard_{args.shard_id:03d}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved={summary_path}", flush=True)


if __name__ == "__main__":
    main()
