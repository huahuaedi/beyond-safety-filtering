"""Aggregate and validate pseudo-control sidecars."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-scenes", type=int, default=7171)
    args = parser.parse_args()
    files = sorted(args.root.glob("*/*.npz"))
    if len(files) != args.expected_scenes:
        raise SystemExit(f"expected {args.expected_scenes} scene files, found {len(files)}")
    total = valid = dynamics = reconstruction = 0
    position, heading = [], []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "initial_state", "pseudo_controls", "reconstructed_trajectory",
                "position_rmse", "heading_rmse", "dynamics_feasible",
                "reconstruction_valid", "control_supervision_valid",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing {sorted(missing)}")
            n = len(data["control_supervision_valid"])
            if data["initial_state"].shape != (n, 5) or data["pseudo_controls"].shape != (n, 8, 2):
                raise ValueError(f"{path}: invalid control shapes")
            if not np.isfinite(data["pseudo_controls"]).all():
                raise ValueError(f"{path}: non-finite controls")
            total += n
            valid += int(data["control_supervision_valid"].sum())
            dynamics += int(data["dynamics_feasible"].sum())
            reconstruction += int(data["reconstruction_valid"].sum())
            position.append(data["position_rmse"])
            heading.append(data["heading_rmse"])
    position, heading = np.concatenate(position), np.concatenate(heading)
    report = {
        "scenes": len(files), "candidates": total,
        "dynamics_feasible": dynamics,
        "reconstruction_valid": reconstruction,
        "control_supervision_valid": valid,
        "valid_fraction": valid / total,
        "position_rmse": {"mean": float(position.mean()), "p50": float(np.quantile(position, .5)), "p95": float(np.quantile(position, .95))},
        "heading_rmse": {"mean": float(heading.mean()), "p50": float(np.quantile(heading, .5)), "p95": float(np.quantile(heading, .95))},
    }
    print(json.dumps(report, indent=2))
    (args.root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("validation: OK")


if __name__ == "__main__":
    main()

