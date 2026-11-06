"""Token-level counterpart of extract_policy_vlm_context.py: the VLM-stream hidden state of every VISUAL token (after all shared layers),
projected to 128 dimensions with a fixed random projection, for Stage A (candidate-conditioned readout of the policy's own visual tokens).

Smoke mode (--smoke, few scenes) prints the sequence layout (how many visual tokens, frames, rows, row widths, any foreign token inside a row),
the cost per scene, the bytes per scene and the extrapolated time and storage for the pool and the navtest split, and writes nothing unless
--output is given. Production runs should pass --expect-grid ROWSxCOLS so that a layout different from the smoke result stops the job.
Layout found by the smoke test: 2 frames, each ONE contiguous run of 576 visual tokens (the video prompt has no eol between rows); a frame is
the row-major flattening of an 18 x 32 grid. Writes <output>/tokens_shard<k>.npz with tokens [N], grid [1152, 3] (frame, row, col; identical for
every scene), h [N, 1152, 128] float16. Frame order follows the dataset's chronological frame selection (last = current); the evaluator does not rely on it.
No simulator, no privileged input. The scene vector never depends on the action noise (checked in extract_policy_vlm_context.py).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import train_offline_verified_policy as T
from offline_verified_data import UNCONDITIONAL
from vlm_token_layout import grid_positions, require_frames, frame_grid, random_projection, projection_fingerprint


class Tap:
    """Wraps the last shared layer (a plain class, not an nn.Module, so no forward hooks) and keeps the VLM-stream output."""

    def __init__(self, layer):
        self.layer, self.captured = layer, None

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        self.captured = out[0]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=None)
    ap.add_argument("--scene-pkl", type=Path, required=True)
    ap.add_argument("--tokens-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="print layout and cost; stop after --limit scenes")
    ap.add_argument("--expect-frames", type=int, default=2, help="number of frames in the video prompt (smoke result: 2)")
    ap.add_argument("--frame-grid", default="18x32", help="HEIGHTxWIDTH of one frame's flattened token run (row-major); smoke result: 18x32")
    ap.add_argument("--no-check", action="store_true", help="smoke only: print the layout without requiring it")
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()
    tokens = json.loads(args.tokens_json.read_text())
    if args.limit:
        tokens = tokens[:args.limit]
    tokens = tokens[args.shard_id::args.num_shards]
    core = T.core
    core.SCENE_PKL = args.scene_pkl
    policy, heads, _, dataset, index, _, _ = T.build_model_dataset(args.base or core.EMU_HUB)
    missing = [t for t in tokens if t not in index]
    if missing:
        raise ValueError(f"{len(missing)} scenes missing from {args.scene_pkl}, e.g. {missing[0]}")
    tok = dataset.tokenizer
    bov, eov = dataset.bov, dataset.eov
    eol_id, eof_id = tok.encode(tok.eol_token)[0], tok.encode(tok.eof_token)[0]
    tap = Tap(heads.shared_layers[-1])
    heads.shared_layers[-1] = tap
    policy.eval()
    proj = None
    grid = None
    rows, times = [], []
    for i, token in enumerate(tokens):
        sample = dataset[index[token]]
        ids, mask, pre, cmd = T.inputs(sample)
        t0 = time.time()
        with torch.no_grad():
            vlm_h = policy.vlm.model.embed_tokens(ids)
            state = torch.cat([pre.reshape(1, -1), cmd], dim=1)
            indicator = torch.tensor([UNCONDITIONAL], device=core.DEVICE, dtype=torch.long)
            state_embedding = core.conditioned_state_embedding(heads, state, indicator)
            positions = torch.arange(ids.shape[1], device=core.DEVICE).unsqueeze(0)
            z = torch.zeros((1, 8, 3), device=core.DEVICE, dtype=core.DTYPE)
            core.denoise_step_velocity(heads, vlm_h, state_embedding, z, 0.5, ids, mask, positions, ids.shape[1], 1)
            full = tap.captured[0].float()
        attended = mask[0].bool().cpu().numpy()
        ids_np = ids[0].cpu().numpy()[attended]
        pos, summary = grid_positions(ids_np, bov, eov, eol_id, eof_id)
        if i == 0 or args.smoke:
            print(f"scene {token}: sequence length {int(attended.sum())}, hidden width {full.shape[1]}, layout {summary}", flush=True)
        gh, gw = (int(x) for x in args.frame_grid.lower().split("x"))
        if not args.no_check:
            require_frames(summary, args.expect_frames, gh * gw)
        if grid is None:
            grid = frame_grid(pos, gh, gw)
            proj = random_projection(full.shape[1], args.dim, 0)
            print(f"projection {full.shape[1]} -> {args.dim}, fingerprint {projection_fingerprint(proj)}", flush=True)
        elif not np.array_equal(grid, frame_grid(pos, gh, gw)):
            raise ValueError(f"scene {token} has a different visual layout from the first scene")
        vis = full[torch.as_tensor(pos[:, 0], dtype=torch.long, device=full.device)].cpu().numpy()
        if not np.isfinite(vis).all():
            raise FloatingPointError(f"nonfinite hidden state for {token}")
        rows.append((vis @ proj).astype(np.float16))
        times.append(time.time() - t0)
        if (i + 1) % 100 == 0:
            print(f"tokens {i + 1}/{len(tokens)}", flush=True)
    if args.smoke or args.output is None:
        per = float(np.mean(times[1:] if len(times) > 1 else times))
        nbytes = rows[0].nbytes
        n_total = 7171 + 12146
        print(f"SMOKE scenes={len(rows)} seconds/scene={per:.2f} bytes/scene={nbytes} (float16, {rows[0].shape})", flush=True)
        print(f"extrapolation to pool + navtest ({n_total} scenes): {per * n_total / 3600:.1f} GPU hours, {nbytes * n_total / 1e9:.1f} GB", flush=True)
        if args.output is None:
            return
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"tokens_shard{args.shard_id}.npz"
    if out.exists():
        raise FileExistsError(out)
    np.savez_compressed(out, tokens=np.array(tokens), grid=grid, h=np.stack(rows), projection_seed=0,
                        projection_fingerprint=projection_fingerprint(proj))
    print(f"COMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
