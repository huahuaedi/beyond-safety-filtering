"""Evaluate a saved action-expert checkpoint (action_policy.pt) or the unmodified base policy on a scene list.

Uses the evaluator of train_offline_verified_policy.py unchanged (one fixed pseudo-random initial latent per scene,
deterministic 10-step ODE, simulator score of the resulting trajectory), so a checkpoint and the base policy are
scored by exactly the same procedure and the comparison is paired scene by scene.

Environment as for the training jobs (NAVSIM_ENV_PYTHON, NAVSIM_METRIC_CACHE_PATH of the cohort, VLA_* paths).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

import train_offline_verified_policy as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=None, help="base policy directory (default: the SFT checkpoint of the core module)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="action_policy.pt; omit to score the base policy")
    ap.add_argument("--scene-pkl", type=Path, required=True, help="records of the scenes (cohort test.pkl)")
    ap.add_argument("--tokens-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--latent-tag", default="offline_eval_v1", help="tag of the per-scene initial latent; the default reproduces every earlier evaluation, other tags give an independent noise draw")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1, help="evaluate every num-shards-th scene; merge with merge_policy_eval_shards.py")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokens = json.loads(args.tokens_json.read_text())
    if args.limit:
        tokens = tokens[: args.limit]
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must lie in [0, num-shards)")
    tokens = tokens[args.shard_id::args.num_shards]
    T.core.SCENE_PKL = args.scene_pkl                      # build_model_dataset reads this at call time
    policy, heads, trainable, dataset, index, low, high = T.build_model_dataset(args.base or T.core.EMU_HUB)
    missing = [t for t in tokens if t not in index]
    if missing:
        raise ValueError(f"{len(missing)} scenes missing from {args.scene_pkl}, e.g. {missing[0]}")
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        T.load_action_state(policy, state["action_state_dict"])
        print(f"loaded {args.checkpoint}", flush=True)
    out = args.output / "validation"
    out.mkdir(parents=True)
    metrics = T.evaluate(policy, heads, dataset, index, tokens, low, high, out, "unconditional", "navtest_cohort", latent_tag=args.latent_tag)
    (args.output / "summary.json").write_text(json.dumps({
        "label": args.label, "checkpoint": str(args.checkpoint) if args.checkpoint else None, "base": str(args.base or T.core.EMU_HUB),
        "latent_tag": args.latent_tag, "scenes": len(tokens), "shard": [args.shard_id, args.num_shards], "validation": metrics, "complete": True}, indent=2) + "\n")
    print(f"COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
