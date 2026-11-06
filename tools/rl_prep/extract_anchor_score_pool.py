"""Read-only extraction of the per-anchor simulator sub-scores for a set of scenes.

formatted_pdm_score_8192.npy is a 47GB pickled dict: token (hex) -> {"trajectory_scores": [ {metric: [8192]} ]}, the supervision of the
Multi-Anchor head (Emu3QFormer, modeling_emu3.py `_anchor_metric_keys`). This loads it once, keeps the scenes of interest, and writes a
compact float16 array [N, 5, 8192] in the metric order NC, DAC, EP, TTC, comfort, plus a JSON report that checks coverage and the value
structure (binary gates? how many anchors pass NC=DAC=TTC=1?) before any training design depends on it. Needs about 60GB of RAM.
Nothing is written next to existing results; --output and --report must not exist.

Scenes come from --tokens-json (list of tokens) or --pool-root (every <split>/<token>.npz below it, e.g. world_action_full_v1).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort")


def scene_tokens(args):
    if args.tokens_json:
        return sorted(json.loads(args.tokens_json.read_text()))
    return sorted({p.stem for p in args.pool_root.rglob("*.npz")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True, help="formatted_pdm_score_8192.npy")
    ap.add_argument("--tokens-json", type=Path, default=None)
    ap.add_argument("--pool-root", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True, help=".npz with tokens, scores[N,5,8192], keys")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    if bool(args.tokens_json) == bool(args.pool_root):
        raise SystemExit("give exactly one of --tokens-json / --pool-root")
    for p in (args.output, args.report):
        if p.exists():
            raise FileExistsError(p)
    wanted = scene_tokens(args)
    print(f"{len(wanted)} requested scenes; loading {args.scores} (expensive)", file=sys.stderr, flush=True)
    full = np.load(args.scores, allow_pickle=True).item()
    print(f"loaded {len(full)} entries", file=sys.stderr, flush=True)
    found, missing, rows = [], [], []
    for t in wanted:
        v = full.get(t)
        if v is None:
            missing.append(t)
            continue
        ts = v["trajectory_scores"][0]
        arr = np.vstack([np.asarray(ts[k], dtype=np.float32) for k in KEYS])
        if arr.shape != (len(KEYS), 8192) or not np.isfinite(arr).all():
            raise RuntimeError(f"bad score block for {t}: {arr.shape}")
        found.append(t)
        rows.append(arr.astype(np.float16))
    del full
    if not rows:
        raise SystemExit(f"no requested token found in the score file; first requested {wanted[:3]}")
    scores = np.stack(rows)                                                   # [N, 5, 8192]
    f32 = scores.astype(np.float32)
    nc, dac, ep, ttc, comf = (f32[:, i] for i in range(5))
    safe = (nc == 1) & (dac == 1) & (ttc == 1)
    n_safe = safe.sum(1)
    report = {
        "requested": len(wanted), "found": len(found), "missing": len(missing), "missing_examples": missing[:5],
        "keys": list(KEYS), "shape": list(scores.shape),
        "value_range": {k: [float(f32[:, i].min()), float(f32[:, i].max())] for i, k in enumerate(KEYS)},
        "distinct_values_first_scene": {k: int(len(np.unique(f32[0, i]))) for i, k in enumerate(KEYS)},
        "binary_gates": {KEYS[i]: bool(np.isin(np.unique(f32[:, i]), [0.0, 1.0]).all()) for i in (0, 1, 3)},
        "safe_anchor_count": {"mean": float(n_safe.mean()), "median": float(np.median(n_safe)),
                              "p05": float(np.percentile(n_safe, 5)), "min": int(n_safe.min()),
                              "scenes_with_0": int((n_safe == 0).sum()), "scenes_with_lt2": int((n_safe < 2).sum()),
                              "scenes_with_ge8": int((n_safe >= 8).sum())},
        "float16_note": "scores stored as float16; EP/comfort lose ~3 decimal digits",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tokens=np.array(found), scores=scores, keys=np.array(KEYS))
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
