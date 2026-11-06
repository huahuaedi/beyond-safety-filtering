"""Candidates from the anchor-scoring planner (Emu3QFormer with anchors): its top-K anchor trajectories per scene.

The model scores 8192 anchor trajectories and deploys the arg-max. Here the K highest-scoring anchors are scored by the
simulator and stored in the format of the sampled-candidate collections, so the selector pipeline applies unchanged. Candidate 0 is
the deployed (arg-max) trajectory; both reference probes are that trajectory, so a gain over the "reference" is a gain over the
model's own output scored by the same worker. The anchor log-probabilities are written to a separate feature root.

Run in the drivevla env with PYTHONPATH containing <repo>/inference/vla (config.py), <repo>/utils and <repo>/reference/Emu3.
Env: VLA_ACTION_TOKENIZER, VLA_VLM_MODEL, VLA_NORM_STATS, VLA_ANCHOR_ENABLED=true, VLA_ANCHOR_CLUSTER_PATH,
     NAVSIM_ENV_PYTHON, NAVSIM_METRIC_CACHE_PATH.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SUB_SCORE_KEYS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
                  "comfort", "driving_direction_compliance")


def stable_scene_split(token):
    bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") % 100
    return "train" if bucket < 80 else ("val" if bucket < 90 else "test")


def score_with_worker(worker, cache, token, trajectories):
    with tempfile.TemporaryDirectory() as tmp:
        np.save(Path(tmp) / "t.npy", trajectories.astype(np.float32))
        res = subprocess.run([os.environ["NAVSIM_ENV_PYTHON"], str(worker), "--metric-cache-path", str(cache), "--scene-token", token,
                              "--trajectories-npy", str(Path(tmp) / "t.npy"), "--output-json", str(Path(tmp) / "s.json")],
                             capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(f"scoring failed for {token}: {res.stderr[-1500:]}")
        payload = json.loads((Path(tmp) / "s.json").read_text())
    rows = payload.get("results", [])
    if "error" in payload or len(rows) != len(trajectories) or any("error" in r for r in rows):
        raise RuntimeError(f"invalid scoring batch for {token}: {payload}")
    scores = {k: np.array([r[k] for r in rows], dtype=np.float32) for k in SUB_SCORE_KEYS}
    if not all(np.isfinite(v).all() for v in scores.values()):
        raise RuntimeError(f"non-finite scores for {token}")
    return scores


@torch.no_grad()
def anchor_scores(model, batch):
    """Logits of all anchors for one scene: the body of Emu3QFormer.sample_actions up to the anchor head."""
    from emu3.mllm import Emu3Pi0
    input_ids, mask = batch["input_ids"], batch["attention_mask"]
    pre_action, cmd = batch["pre_action"].to(torch.bfloat16), batch["cmd"].to(torch.bfloat16)
    b, device = input_ids.shape[0], input_ids.device
    vlm_h = model.vlm.model.embed_tokens(input_ids)
    vlm_len = vlm_h.shape[1]
    state = model.state_projector(torch.cat([pre_action.view(b, -1), cmd], dim=1)).unsqueeze(1)
    action_h = torch.cat([state, model.action_queries.expand(b, -1, -1)], dim=1)
    positions = torch.arange(vlm_len, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)
    mask4d = Emu3Pi0.create_causal_style_attention_mask(vlm_len, action_h.shape[1], mask, input_ids, b, device, vlm_h.dtype)
    for layer in model.shared_layers[: len(model.vlm.model.layers)]:
        vlm_h, action_h = layer(vlm_h, action_h, positions, mask4d, vlm_len, action_h.shape[1], b)
    hidden = model.action_expert.norm(action_h)
    feat, _ = model._get_cluster_centers_feat(b)
    return model.anchor_policy_head(model.anchor_tf_decoder(feat, hidden)).squeeze(-1)           # [b, 8192]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emu-hub", type=Path, required=True)
    ap.add_argument("--scene-pkl", type=Path, required=True)
    ap.add_argument("--tokens-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="collection root; anchor features go to <output>_anchor_feat")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    repo = Path(os.environ.get("DRIVEVLA_REPO_ROOT", "."))
    for sub in ("inference/vla", "utils", "reference/Emu3", "tools/rl_prep"):
        sys.path.insert(0, str(repo / sub))
    from config import setup_paths_early
    setup_paths_early()
    from emu3.mllm import Emu3Tokenizer, Emu3QFormer, Emu3Pi0Config
    from utils.train_qformer import DataArguments
    from datasets import Emu3DrivingVAVADataset

    import emu3.mllm.modeling_emu3 as modeling
    print("modeling_emu3 from:", modeling.__file__, flush=True)
    # The released __init__ assigns self.anchor before self._anchor_setup_done exists, so the property setter fails whenever
    # the anchor head is enabled. A class-level default makes that early assignment only record the flag; __init__ later sets the
    # instance value to False and builds the anchor modules (so their weights are loaded) in its own `if self.anchor:` call.
    Emu3QFormer._anchor_setup_done = True
    device = "cuda"
    data_args = DataArguments(data_path=str(args.scene_pkl), actions=True, driving=True, use_previous_actions=True,
                              actions_format="fast", action_tokenizer_path=os.environ["VLA_ACTION_TOKENIZER"], frames=1,
                              action_frames=8, action_dim=3, cur_frame_idx=3, pre_action_frames=3, video_format=None, use_flip=False)
    tokenizer = Emu3Tokenizer.from_pretrained(os.environ["VLA_VLM_MODEL"], model_max_length=1400, padding_side="right", use_fast=False)
    dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    index = {s["token"]: i for i, s in enumerate(dataset.data)}
    config = Emu3Pi0Config.from_pretrained(str(args.emu_hub / "config.json"))
    model, info = Emu3QFormer.from_pretrained(str(args.emu_hub), config=config, pretrain_vlm_path=os.environ["VLA_VLM_MODEL"],
                                              attn_implementation="sdpa", torch_dtype=torch.bfloat16, trust_remote_code=True,
                                              output_loading_info=True)
    print("missing", info["missing_keys"][:5], "unexpected", info["unexpected_keys"][:5], flush=True)
    model = model.to(device).eval()
    model._ensure_anchor_setup()
    if not getattr(model, "anchor", False):
        raise SystemExit("model has anchor=False; set VLA_ANCHOR_ENABLED=true")
    clusters = model.anchor_cluster_centers.to(device)
    worker = repo / "tools/rl_prep/navsim_score_worker.py"
    cache = os.environ["NAVSIM_METRIC_CACHE_PATH"]
    tokens = json.loads(args.tokens_json.read_text())[args.shard_id::args.num_shards]
    if args.limit:
        tokens = tokens[: args.limit]
    feat_root = Path(str(args.output) + "_anchor_feat")
    done = 0
    for token in tokens:
        split = stable_scene_split(token)
        out = args.output / split / f"{token}.npz"
        if out.exists():
            continue
        sample = dataset[index[token]]
        batch = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample.items()
                 if k in ("input_ids", "attention_mask", "pre_action", "cmd")}
        logits = anchor_scores(model, batch)[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        top = torch.topk(logp, args.k)
        trajectories = clusters[top.indices].float().cpu().numpy()                       # [K, 8, 3] absolute SE2
        scores = score_with_worker(worker, cache, token, trajectories)
        relative = np.stack([np.diff(np.concatenate([np.zeros((1, 3)), t], axis=0), axis=0) for t in trajectories])
        expert = sample["action"].detach().float().cpu().numpy() if isinstance(sample["action"], torch.Tensor) else np.asarray(sample["action"])
        if expert.shape != (8, 3):
            raise ValueError(f"expert action shape {expert.shape} for {token}")
        payload = {"schema_version": np.asarray(1, dtype=np.int32), "scene_token": np.asarray(token), "split": np.asarray(split),
                   "relative_trajectories": relative.astype(np.float32), "absolute_trajectories": trajectories.astype(np.float32),
                   "expert_action_normalized": expert.astype(np.float32), "candidate_seed": np.asarray(0, dtype=np.int64),
                   "reference_probe_absolute_trajectories": np.stack([trajectories[0]] * 2).astype(np.float32)}
        for key in SUB_SCORE_KEYS:
            payload[f"score__{key}"] = scores[key]
            payload[f"reference_probe__{key}"] = np.array([scores[key][0]] * 2, dtype=np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out.with_suffix(".tmp.npz"), **payload)
        os.replace(out.with_suffix(".tmp.npz"), out)
        feats = np.stack([top.values.cpu().numpy(), (top.values - top.values[0]).cpu().numpy()], axis=1).astype(np.float32)
        (feat_root / split).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feat_root / split / f"{token}.npz", features=feats)
        done += 1
        if done % 25 == 0 or done == 1:
            print(f"{done}/{len(tokens)} scenes written", flush=True)
    print(json.dumps({"written": done, "shard": [args.shard_id, args.num_shards]}))


if __name__ == "__main__":
    main()
