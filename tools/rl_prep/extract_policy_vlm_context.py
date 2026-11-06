"""Scene-level representation that the policy itself conditions on: the VLM-stream hidden state after all shared layers, mean-pooled over
the attended tokens (the vector returned by denoise_step_velocity(..., return_vlm_context=True), also exported by the released inference
with --export_vlm_context). The prefix cannot attend to the action tokens, so the vector depends only on the scene input and not on the
candidate, the noise or the flow time; the first scene is re-run with two different noises to verify this.

Writes <output>/context_shard<k>.npz with tokens [N] and contexts [N, D] (float16). No simulator, no privileged information.
Usage (GPU, environment as for eval_action_policy.py):
  extract_policy_vlm_context.py --scene-pkl P --tokens-json T --output DIR [--shard-id k --num-shards n]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_offline_verified_policy as T
from offline_verified_data import UNCONDITIONAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=None)
    ap.add_argument("--scene-pkl", type=Path, required=True)
    ap.add_argument("--tokens-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    tokens = json.loads(args.tokens_json.read_text())
    if args.limit:
        tokens = tokens[:args.limit]
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must lie in [0, num-shards)")
    tokens = tokens[args.shard_id::args.num_shards]
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"context_shard{args.shard_id}.npz"
    if out.exists():
        raise FileExistsError(out)
    core = T.core
    core.SCENE_PKL = args.scene_pkl
    policy, heads, _, dataset, index, _, _ = T.build_model_dataset(args.base or core.EMU_HUB)
    missing = [t for t in tokens if t not in index]
    if missing:
        raise ValueError(f"{len(missing)} scenes missing from {args.scene_pkl}, e.g. {missing[0]}")
    policy.eval()
    rows = []

    def context(sample, seed):
        ids, mask, pre, cmd = T.inputs(sample)
        with torch.no_grad():
            vlm_h = policy.vlm.model.embed_tokens(ids)
            state = torch.cat([pre.reshape(1, -1), cmd], dim=1)
            indicator = torch.tensor([UNCONDITIONAL], device=core.DEVICE, dtype=torch.long)
            state_embedding = core.conditioned_state_embedding(heads, state, indicator)
            positions = torch.arange(ids.shape[1], device=core.DEVICE).unsqueeze(0)
            gen = torch.Generator(device=core.DEVICE).manual_seed(seed)
            z = torch.randn((1, 8, 3), generator=gen, device=core.DEVICE, dtype=core.DTYPE)
            _, pooled = core.denoise_step_velocity(heads, vlm_h, state_embedding, z, 0.5, ids, mask, positions, ids.shape[1], 1,
                                                   return_vlm_context=True)
        return pooled[0].float().cpu().numpy()

    for i, token in enumerate(tokens):
        sample = dataset[index[token]]
        v = context(sample, 0)
        if i == 0:
            w = context(sample, 1)
            rel = float(np.abs(v - w).max() / (np.abs(v).max() + 1e-9))
            print(f"noise-independence check: max relative difference {rel:.2e}", flush=True)
            if rel > 1e-2:
                raise RuntimeError("the pooled VLM context depends on the action noise; it is not a scene-only representation")
        if not np.isfinite(v).all():
            raise FloatingPointError(f"nonfinite context for {token}")
        rows.append(v.astype(np.float16))
        if (i + 1) % 100 == 0 or i == 0:
            print(f"context {i + 1}/{len(tokens)}", flush=True)
    np.savez_compressed(out, tokens=np.array(tokens), contexts=np.stack(rows))
    print(f"COMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
