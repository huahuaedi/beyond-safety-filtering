"""Approximation error eps_w of Prop. 3 on the training scenes: distance between the fitted policy's output and the selected winner.

For each pair in --pairs (a random subsample of --scenes), the policy (base or --checkpoint) generates a trajectory from the
same fixed per-scene initial latent and deterministic 10-step ODE as the evaluator, and is compared with the winner's relative
trajectory read from the candidate data. Distances are computed (i) in the normalized action space, over all 24 dims, and (ii) in
metres over the x,y waypoints. Run for the base policy (omit --checkpoint) and for a fitted checkpoint with the SAME pairs file
to get the error before and after fitting. Uses the default (trainval) scene pkl, as training did. Needs a GPU and the training env.
Output: JSON with per-scene values and means; eps = sqrt(mean squared distance).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import train_offline_verified_policy as T
from offline_verified_data import read_scene, normalize_action, stable_seed
core = T.core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True, help="pairs.jsonl (token, split, winner, loser)")
    ap.add_argument("--source-root", type=Path, required=True, help="candidate data root (world_action_full_v1)")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--base", type=Path, default=None)
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--latent-tag", default="offline_eval_v1")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    pairs = sorted(pairs, key=lambda p: p["token"])
    rng = np.random.default_rng(0)
    if args.scenes and args.scenes < len(pairs):
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), args.scenes, replace=False))]
    policy, heads, trainable, dataset, index, low, high = T.build_model_dataset(args.base or core.EMU_HUB)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        T.load_action_state(policy, state["action_state_dict"])
    policy.eval()
    rows = []
    for i, p in enumerate(pairs):
        token = p["token"]
        relative, _, _ = read_scene(args.source_root / p["split"] / f"{token}.npz")
        win_rel = relative[p["winner"]]
        ids, mask, pre, cmd = T.inputs(dataset[index[token]])
        with torch.no_grad():
            gen = torch.Generator(device=core.DEVICE).manual_seed(stable_seed(token, args.latent_tag))
            initial = torch.randn((1, 8, 3), generator=gen, device=core.DEVICE, dtype=core.DTYPE)
            action, _, _ = core.rollout_and_score(policy, heads, ids, mask, pre, cmd, 8, 3, 10, 0.,
                initial_z=initial, advantage_indicator=torch.tensor([T.UNCONDITIONAL], device=core.DEVICE))
        out_norm = action[0].float().cpu().numpy()
        out_rel = .5 * (out_norm + 1) * (high - low) + low
        win_norm = normalize_action(win_rel, low, high)
        d_norm = float(((out_norm - win_norm) ** 2).sum())
        d_xy = float(np.linalg.norm(out_rel[:, :2] - win_rel[:, :2], axis=-1).mean())
        rows.append({"token": token, "sq_dist_normalized": d_norm, "mean_xy_error_m": d_xy})
        if (i + 1) % 25 == 0 or i == 0:
            print(f"{i+1}/{len(pairs)} sq={d_norm:.4f} xy={d_xy:.3f}", flush=True)
    sq = np.array([r["sq_dist_normalized"] for r in rows]); xy = np.array([r["mean_xy_error_m"] for r in rows])
    summary = {"pairs_file": str(args.pairs), "checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "latent_tag": args.latent_tag, "scenes": len(rows), "eps_normalized": float(np.sqrt(sq.mean())),
               "mean_xy_error_m": float(xy.mean()), "median_xy_error_m": float(np.median(xy)), "rows": rows}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
