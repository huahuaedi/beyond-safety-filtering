"""Build compact quality-conditioning labels beside world-action rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quality_conditioning import RISK_NAMES, build_quality_metadata


METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
)
SAFETY_METRICS = {
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
}


def load_weights(path: Path | None):
    if path is None:
        return {}
    with path.open() as handle:
        manifest = json.load(handle)
    result = {}
    for entry in manifest.get("entries", []):
        token = str(entry["scene_token"])
        weights = entry.get("dynamic_constraint_weights")
        if weights is not None:
            result[token] = {name: float(weights[name]) for name in RISK_NAMES}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--weight-manifest", type=Path)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("require 0 <= shard-id < num-shards")

    files = sorted(args.input_root.glob("*/*.npz"))
    files = files[args.shard_id :: args.num_shards]
    if not files:
        raise SystemExit(f"no input files assigned from {args.input_root}")
    weight_map = load_weights(args.weight_manifest)
    written = skipped = 0

    for index, source_path in enumerate(files):
        relative = source_path.relative_to(args.input_root)
        output_path = args.output_root / relative
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        with np.load(source_path, allow_pickle=False) as data:
            token = str(data["scene_token"])
            scores = {name: data[f"score__{name}"].astype(np.float32) for name in METRICS}
            reference = {}
            for name in METRICS:
                probes = data[f"reference_probe__{name}"].astype(np.float32)
                reference[name] = float(probes.max() if name in SAFETY_METRICS else probes.mean())
            weights = weight_map.get(token, {name: 1.0 for name in RISK_NAMES})
            metadata = build_quality_metadata(scores, reference, weights)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".tmp")
        payload = {
            "schema_version": np.asarray(1, dtype=np.int32),
            "scene_token": np.asarray(token),
            "split": np.asarray(source_path.parent.name),
            **metadata,
            "constraint_weights": np.asarray([weights[name] for name in RISK_NAMES], dtype=np.float32),
        }
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(output_path)
        written += 1
        if index % 250 == 0 or index + 1 == len(files):
            print(f"quality {index + 1}/{len(files)} scene={token} path={output_path}", flush=True)

    print(
        f"quality sidecars complete assigned={len(files)} written={written} skipped={skipped} "
        f"output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
