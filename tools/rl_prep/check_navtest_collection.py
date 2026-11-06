"""Format and content check of a candidate collection; run before the full collection.

Usage: check_navtest_collection.py <collection_dir> [<test_tokens.json>]
With the token file it also requires every collected scene to belong to that set, and it rejects
collections whose scores look like failed scoring (all zeros / non-finite / no progress signal).
"""
import json
import sys
from pathlib import Path
import numpy as np

REQUIRED = {"relative_trajectories": (None, 8, 3), "absolute_trajectories": (None, 8, 3),
            "expert_action_normalized": (8, 3), "reference_probe_absolute_trajectories": (2, 8, 3)}
SCORES = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
          "ego_progress", "comfort", "driving_direction_compliance")


def check(path):
    problems = []
    with np.load(path, allow_pickle=False) as z:
        for k, shape in REQUIRED.items():
            if k not in z.files:
                problems.append(f"missing {k}"); continue
            if any(a is not None and a != b for a, b in zip(shape, z[k].shape)):
                problems.append(f"{k} shape {z[k].shape}")
        n = z["relative_trajectories"].shape[0] if "relative_trajectories" in z.files else 0
        for m in SCORES:
            if f"score__{m}" not in z.files or z[f"score__{m}"].shape != (n,):
                problems.append(f"score__{m}")
            if f"reference_probe__{m}" not in z.files or z[f"reference_probe__{m}"].shape != (2,):
                problems.append(f"reference_probe__{m}")
            elif not np.isfinite(z[f"reference_probe__{m}"]).all():
                problems.append(f"nonfinite reference_probe__{m}")
        if not problems:
            if not all(np.isfinite(z[f"score__{m}"]).all() for m in SCORES):
                problems.append("nonfinite candidate scores")
            elif float(z["score__ego_progress"].max()) <= 0.0 and float(z["reference_probe__ego_progress"].max()) <= 0.0:
                problems.append("no progress signal at all: scoring probably failed")
    return n, problems


if __name__ == "__main__":
    files = [p for s in ("train", "val", "test") for p in sorted((Path(sys.argv[1]) / s).glob("*.npz"))]
    if not files:
        raise SystemExit("no collected scenes found")
    bad = 0
    if len(sys.argv) > 2:
        allowed = set(json.load(open(sys.argv[2])))
        stray = [p.stem for p in files if p.stem not in allowed]
        print(f"{len(files)} collected scenes, {len(allowed)} allowed tokens, {len(stray)} outside the allowed set")
        if stray:
            raise SystemExit(f"collection contains scenes outside {sys.argv[2]}, e.g. {stray[:3]}")
    # A scene in which every trajectory leaves the drivable area (DAC = 0, hence progress = 0 for all candidates and probes) is
    # a legitimate hard scene, not a scoring failure. It only counts as a failure signal when it dominates the collection.
    SOFT = "no progress signal at all: scoring probably failed"
    soft = 0
    for p in files:
        n, problems = check(p)
        hard = [q for q in problems if q != SOFT]
        soft += SOFT in problems
        print(p.name, "candidates", n, "OK" if not problems else problems)
        bad += bool(hard)
    print(f"{soft} of {len(files)} scenes have no progress signal (all trajectories off the drivable area or scoring failed)")
    if soft > 0.5 * len(files):
        print("more than half of the scenes have no progress signal: scoring probably failed")
        bad += 1
    raise SystemExit(1 if bad else 0)
