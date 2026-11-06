"""Validate sharded world-action candidate data before latent extraction."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "scene_token",
    "split",
    "relative_trajectories",
    "absolute_trajectories",
    "reference_feasible",
    "reference_violation",
    "score__no_at_fault_collisions",
    "score__drivable_area_compliance",
    "score__ego_progress",
    "score__time_to_collision_within_bound",
    "score__comfort",
    "score__driving_direction_compliance",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    files = sorted(args.root.glob("*/*.npz"))
    if not files:
        raise SystemExit(f"no split/*.npz files found under {args.root}")

    split_counts = Counter()
    candidate_counts = Counter()
    tokens = set()
    metric_values = {key: [] for key in REQUIRED_KEYS if key.startswith("score__")}
    total_candidates = total_unsafe = 0

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            missing = REQUIRED_KEYS - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing keys {sorted(missing)}")
            token = str(data["scene_token"])
            split = str(data["split"])
            if split != path.parent.name:
                raise ValueError(f"{path}: embedded split={split!r}")
            if token in tokens:
                raise ValueError(f"duplicate scene token: {token}")
            tokens.add(token)

            rel = data["relative_trajectories"]
            absolute = data["absolute_trajectories"]
            if rel.ndim != 3 or rel.shape[1:] != (8, 3) or absolute.shape != rel.shape:
                raise ValueError(f"{path}: bad trajectory shapes {rel.shape}, {absolute.shape}")
            if not np.isfinite(rel).all() or not np.isfinite(absolute).all():
                raise ValueError(f"{path}: non-finite trajectory")

            group_size = rel.shape[0]
            feasible = data["reference_feasible"]
            violation = data["reference_violation"]
            if feasible.shape != (group_size,) or violation.shape != (group_size,):
                raise ValueError(f"{path}: label shape does not match group size")
            if not np.isfinite(violation).all():
                raise ValueError(f"{path}: non-finite violation")

            for key in metric_values:
                values = data[key]
                if values.shape != (group_size,) or not np.isfinite(values).all():
                    raise ValueError(f"{path}: invalid {key} shape/values")
                metric_values[key].append(values.astype(np.float64))

            split_counts[split] += 1
            candidate_counts[group_size] += 1
            total_candidates += group_size
            total_unsafe += int((~feasible).sum())

    print(f"root: {args.root}")
    print(f"scenes: {len(files)}; candidates: {total_candidates}")
    print(f"scene splits: {dict(sorted(split_counts.items()))}")
    print(f"candidate groups: {dict(sorted(candidate_counts.items()))}")
    print(f"reference-relative unsafe fraction: {total_unsafe / total_candidates:.6f}")
    for key, chunks in sorted(metric_values.items()):
        values = np.concatenate(chunks)
        print(
            f"{key.removeprefix('score__')}: mean={values.mean():.6f} "
            f"failure_rate={(values < 1.0 - 1e-8).mean():.6f}"
        )
    print("validation: OK")


if __name__ == "__main__":
    main()

