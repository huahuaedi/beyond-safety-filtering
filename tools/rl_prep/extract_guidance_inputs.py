"""Per-scene inputs of the geometric guidance cost, from the NAVSIM metric cache (navsim env, CPU).

For every scene token it writes <output-root>/<token>.npz with, in the ego frame at t=0 (rear axle at the origin, x forward):
the agents observed at t=0 (position, velocity, heading, length, width), a signed-distance raster of the drivable area
(metres, positive inside) with its origin and resolution, and the ego footprint. No logged future of other agents is used.

Usage: python extract_guidance_inputs.py --tokens-json T --metric-cache-path M --output-root O [--limit K]
"""
import argparse
import json
import lzma
import os
import pickle
import sys
from pathlib import Path

import numpy as np

X_RANGE = (-10.0, 60.0)
Y_RANGE = (-30.0, 30.0)
RES = 0.25


def global_to_ego(points, origin):
    ox, oy, oh = origin
    c, s = np.cos(oh), np.sin(oh)
    d = np.asarray(points, dtype=np.float64) - np.array([ox, oy])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)


def ego_to_global(points, origin):
    ox, oy, oh = origin
    c, s = np.cos(oh), np.sin(oh)
    p = np.asarray(points, dtype=np.float64)
    return np.stack([c * p[..., 0] - s * p[..., 1] + ox, s * p[..., 0] + c * p[..., 1] + oy], axis=-1)


def build_sdf(inside_global, origin, x_range=X_RANGE, y_range=Y_RANGE, res=RES):
    """Signed distance (m, positive inside the drivable area) on an ego-frame grid; rows follow y, columns follow x."""
    from scipy.ndimage import distance_transform_edt
    xs = np.arange(x_range[0], x_range[1] + 1e-9, res)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, res)
    gx, gy = np.meshgrid(xs, ys)
    pts = ego_to_global(np.stack([gx, gy], axis=-1).reshape(-1, 2), origin)
    mask = np.asarray(inside_global(pts), dtype=bool).reshape(gx.shape)
    if mask.all():
        sdf = np.full(mask.shape, 50.0)
    elif not mask.any():
        sdf = np.full(mask.shape, -50.0)
    else:
        sdf = (distance_transform_edt(mask) - distance_transform_edt(~mask)) * res
    return sdf.astype(np.float16), (float(xs[0]), float(ys[0])), float(res)


def agents_in_ego_frame(agents, origin):
    if agents is None:
        z = np.zeros((0, 2))
        return {"pos": z, "vel": z, "heading": np.zeros(0), "length": np.zeros(0), "width": np.zeros(0)}
    ox, oy, oh = origin
    c, s = np.cos(oh), np.sin(oh)
    vel = np.asarray(agents["vel"], dtype=np.float64)
    return {"pos": global_to_ego(agents["pos"], origin),
            "vel": np.stack([c * vel[:, 0] + s * vel[:, 1], -s * vel[:, 0] + c * vel[:, 1]], axis=-1),
            "heading": np.asarray(agents["heading"], dtype=np.float64) - oh,
            "length": np.asarray(agents["length"]), "width": np.asarray(agents["width"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-json", type=Path, required=True)
    ap.add_argument("--metric-cache-path", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import extract_geometric_features as E
    from navsim.common.dataloader import MetricCacheLoader

    loader = MetricCacheLoader(args.metric_cache_path)
    tokens = json.loads(args.tokens_json.read_text())[args.shard_id::args.num_shards]
    if args.limit:
        tokens = tokens[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    done = 0
    for token in tokens:
        out = args.output_root / f"{token}.npz"
        if out.exists():
            continue
        if token not in loader.metric_cache_paths:
            raise SystemExit(f"no metric cache for {token}")
        mc = E.load_metric_cache(loader.metric_cache_paths[token])
        rear = mc.ego_state.rear_axle
        origin = (float(rear.x), float(rear.y), float(rear.heading))
        agents, _, _ = E.agents_from_metric_cache(mc)
        sdf, sdf_origin, res = build_sdf(E.drivable_predicate(mc), origin)
        dims = E.ego_dimensions(mc.ego_state)
        ae = agents_in_ego_frame(agents, origin)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, sdf=sdf, sdf_origin=np.array(sdf_origin), sdf_res=np.array(res),
                            ego_length=dims["length"], ego_width=dims["width"], ego_center_offset=dims["center_offset"],
                            agent_pos=ae["pos"], agent_vel=ae["vel"], agent_heading=ae["heading"],
                            agent_length=ae["length"], agent_width=ae["width"], scene_token=np.asarray(token))
        os.replace(tmp, out)
        done += 1
        if done % 100 == 0:
            print(f"{done} scenes written", flush=True)
    print(json.dumps({"written": done, "shard": [args.shard_id, args.num_shards]}))


if __name__ == "__main__":
    main()
