"""Extract non-learned geometric risk features for every collected candidate.

Runs in the ``navsim`` conda env (needs nuPlan/NAVSIM to unpickle metric
caches).  For each ``<data-root>/<split>/<token>.npz`` it writes a matching
``<output-root>/<split>/<token>.npz`` holding ``geom_features`` [64, F] whose
rows follow the candidate order of ``absolute_trajectories``.

Only information available to a deployed perception stack is used: agent boxes
and velocities at t=0 (objects that first appear later in the logged future are
ignored), the drivable-area polygons, and the candidate itself.  The logged
future of other agents -- which defines the NAVSIM labels -- is never read.

Usage:
    python extract_geometric_features.py --data-root D --metric-cache-path M \
        --output-root O [--shard-id I --num-shards N] [--limit K]
"""
import argparse
import json
import lzma
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometric_features as gf  # noqa: E402
import perception_noise as pn  # noqa: E402

SPLITS = ("train", "val", "test")


def ego_dimensions(ego_state):
    params = ego_state.car_footprint.vehicle_parameters
    length = float(getattr(params, "length", params.front_length + params.rear_length))
    width = float(params.width)
    center, rear = ego_state.center, ego_state.rear_axle
    offset = float(np.hypot(center.x - rear.x, center.y - rear.y))
    return {"length": length, "width": width, "center_offset": offset}


def agents_from_metric_cache(metric_cache):
    """Agents present at t=0 with their current box and velocity (global frame)."""
    observation = metric_cache.observation
    present = set(observation[0].tokens)
    pos, vel, heading, length, width = [], [], [], [], []
    moving = 0
    for token, obj in observation.unique_objects.items():
        if token not in present:
            continue  # first seen in the logged future: not observable now
        box = obj.box
        velocity = getattr(obj, "velocity", None)
        vx, vy = (float(velocity.x), float(velocity.y)) if velocity is not None else (0.0, 0.0)
        moving += int(np.hypot(vx, vy) > 0.2)
        pos.append((float(box.center.x), float(box.center.y)))
        vel.append((vx, vy))
        heading.append(float(box.center.heading))
        length.append(float(box.length))
        width.append(float(box.width))
    if not pos:
        return None, 0, 0
    agents = {
        "pos": np.asarray(pos), "vel": np.asarray(vel), "heading": np.asarray(heading),
        "length": np.asarray(length), "width": np.asarray(width),
    }
    return agents, len(pos), moving


def drivable_predicate(metric_cache):
    drivable = metric_cache.drivable_area_map

    def inside(points):
        return np.asarray(drivable.points_in_polygons(points)).any(axis=0)

    return inside


def load_metric_cache(path):
    with lzma.open(path, "rb") as handle:
        return pickle.load(handle)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--metric-cache-path", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="process at most this many scenes (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--noise-level", type=int, default=0, choices=sorted(pn.LEVELS),
                    help="synthetic perception errors on agents and map (0 = clean); see perception_noise.py")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--noise-target", choices=["both", "map", "agents"], default="both",
                    help="which part of the scene state receives the synthetic errors (default both: agents and the map offset)")
    ap.add_argument("--ttc", action="store_true",
                    help="write the TTC-aware feature block (key 'features') instead of the 12 base features")
    args = ap.parse_args(argv)
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit(f"invalid shard {args.shard_id}/{args.num_shards}")

    from navsim.common.dataloader import MetricCacheLoader

    loader = MetricCacheLoader(args.metric_cache_path)
    files = sorted(p for split in SPLITS for p in (args.data_root / split).glob("*.npz"))
    if not files:
        raise SystemExit(f"no scene files under {args.data_root}")
    files = files[args.shard_id::args.num_shards]
    if args.limit:
        files = files[: args.limit]

    missing, done, skipped = [], 0, 0
    agent_total = moving_total = 0
    for path in files:
        out = args.output_root / path.parent.name / path.name
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        with np.load(path, allow_pickle=False) as data:
            token = str(data["scene_token"])
            candidates = data["absolute_trajectories"].astype(np.float64)
        if token != path.stem:
            raise ValueError(f"{path}: embedded scene_token {token!r} != file stem")
        if token not in loader.metric_cache_paths:
            missing.append(token)
            continue
        metric_cache = load_metric_cache(loader.metric_cache_paths[token])
        rear = metric_cache.ego_state.rear_axle
        agents, n_agents, n_moving = agents_from_metric_cache(metric_cache)
        origin = (float(rear.x), float(rear.y), float(rear.heading))
        dims = ego_dimensions(metric_cache.ego_state)
        inside = drivable_predicate(metric_cache)
        if args.noise_level:
            rng = pn.scene_rng(token, args.noise_level, args.noise_seed)
            perturbed = pn.perturb_agents(agents, args.noise_level, rng)   # always drawn, so the map offset is identical in every mode
            offset = pn.map_offset(args.noise_level, rng)
            if args.noise_target in ("both", "agents"):
                agents = perturbed
            if args.noise_target in ("both", "map"):
                inside = pn.shift_predicate(inside, offset)
        if args.ttc:
            features = gf.scene_ttc_features(candidates, origin, agents, dims)
        else:
            features = gf.scene_geometric_features(candidates, origin, agents, dims, inside)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            **{"features" if args.ttc else "geom_features": features},
            feature_names=np.asarray(gf.TTC_FEATURE_NAMES if args.ttc else gf.FEATURE_NAMES),
            scene_token=np.asarray(token),
            num_agents=np.asarray(n_agents),
            num_moving_agents=np.asarray(n_moving),
        )
        os.replace(tmp, out)
        done += 1
        agent_total += n_agents
        moving_total += n_moving
        if done % 100 == 0:
            print(f"shard {args.shard_id}: {done} scenes written", flush=True)

    summary = {
        "shard": [args.shard_id, args.num_shards], "written": done, "skipped_existing": skipped,
        "missing_metric_cache": len(missing), "mean_agents": agent_total / max(done, 1),
        "moving_agent_fraction": moving_total / max(agent_total, 1),
    }
    print(json.dumps(summary), flush=True)
    if agent_total >= 200 and moving_total == 0:
        raise SystemExit(
            "no agent has a non-zero velocity: the metric cache does not carry velocities, so the "
            "constant-velocity forecast is invalid. Fix the adapter before using these features."
        )
    if missing:
        print("missing metric caches (first 20): " + ", ".join(missing[:20]), file=sys.stderr)
        raise SystemExit(f"{len(missing)} scenes have no metric cache; refusing to leave a partial feature set")


if __name__ == "__main__":
    main()
