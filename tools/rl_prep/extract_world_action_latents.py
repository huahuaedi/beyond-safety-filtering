"""Extract action-conditioned NuPlan world-model hidden states.

Consumes the verified split/scene.npz files produced by RL collection and
writes matching latent files. Each candidate is encoded as the same current
visual context followed by its FAST-tokenized action; hidden states are taken
at the end-of-action token from several decoder depths.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "reference" / "Emu3"))

from emu3.mllm import Emu3MoE, Emu3Tokenizer  # noqa: E402

NUPLAN_CKPT = Path(os.environ.get(
    "WM_NUPLAN_CKPT",
    REPO_ROOT / "pretrained_models/DriveVLA-W0/Emu3_NuPlan_Pretrain_Cktps",
))
FAST_TOKENIZER_PATH = REPO_ROOT / "pretrained_models/fast"
NORM_STATS_PATH = REPO_ROOT / "configs/normalizer_navsim_trainval/norm_stats.json"
SCENE_PKL = REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
VQ_ROOT = REPO_ROOT / "data/navsim/processed_data/data/navsim/processed_data"

DATA_ROOT = Path(os.environ.get("WM_ACTION_DATA_ROOT", REPO_ROOT / "navsim_data/exp/world_action_v1"))
LATENT_ROOT = Path(os.environ.get("WM_LATENT_ROOT", REPO_ROOT / "navsim_data/exp/world_action_latents_v1"))
BATCH_SIZE = int(os.environ.get("WM_LATENT_BATCH_SIZE", 8))
LAYER_SPECS = [int(x) for x in os.environ.get("WM_LATENT_LAYERS", "-1,-9,-17,-25").split(",")]
SHARD_ID = int(os.environ.get("WM_LATENT_SHARD_ID", 0))
NUM_SHARDS = int(os.environ.get("WM_LATENT_NUM_SHARDS", 1))
MAX_SCENES = int(os.environ.get("WM_LATENT_MAX_SCENES", 0))
CUR_IDX = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

PATH_PREFIXES = (
    "/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/",
    "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/",
    "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/",
)


def resolve_vq_path(raw_path: str) -> Path:
    for old in PATH_PREFIXES:
        if raw_path.startswith(old):
            return Path(raw_path.replace(old, str(VQ_ROOT) + "/"))
    return Path(raw_path)


def format_video_prompt(tokenizer, vq_tokens: np.ndarray) -> str:
    frames, height, width = vq_tokens.shape
    frame_strings = []
    for frame in vq_tokens:
        rows = [
            "".join(f"<|visual token {token:0>6d}|>" for token in row)
            for row in frame
        ]
        frame_strings.append(tokenizer.eol_token.join(rows))
    video = tokenizer.eof_token.join(frame_strings)
    return (
        tokenizer.boi_token
        + f"{frames}*{height}*{width}"
        + tokenizer.img_token
        + video
        + tokenizer.eof_token
        + tokenizer.eoi_token
    )


def main():
    if NUM_SHARDS < 1 or not 0 <= SHARD_ID < NUM_SHARDS:
        raise ValueError(f"invalid shard {SHARD_ID}/{NUM_SHARDS}")
    files = sorted(DATA_ROOT.glob("*/*.npz"))[SHARD_ID::NUM_SHARDS]
    if MAX_SCENES > 0:
        files = files[:MAX_SCENES]
    if not files:
        raise SystemExit(f"no input files for shard {SHARD_ID}/{NUM_SHARDS} under {DATA_ROOT}")
    required_checkpoint_files = (
        "config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    missing_checkpoint_files = [
        name for name in required_checkpoint_files if not (NUPLAN_CKPT / name).is_file()
    ]
    if missing_checkpoint_files:
        raise FileNotFoundError(
            f"NuPlan world-model checkpoint is incomplete at {NUPLAN_CKPT}; "
            f"missing {missing_checkpoint_files}. Set WM_NUPLAN_CKPT to the real "
            "local directory or run: bash scripts/hpc/fetch_data.sh nuplan_wm"
        )

    print(f"Loading world model from {NUPLAN_CKPT} ...", flush=True)
    tokenizer = Emu3Tokenizer.from_pretrained(
        str(NUPLAN_CKPT), model_max_length=4000, padding_side="right", use_fast=False
    )
    model = Emu3MoE.from_pretrained(
        str(NUPLAN_CKPT), attn_implementation="sdpa", torch_dtype=DTYPE
    ).to(DEVICE).eval()
    action_tokenizer = AutoProcessor.from_pretrained(
        str(FAST_TOKENIZER_PATH), trust_remote_code=True
    )

    with open(NORM_STATS_PATH) as f:
        norm_cfg = json.load(f)["norm_stats"]["libero"]
    low = np.asarray(norm_cfg["q01"], dtype=np.float32)
    high = np.asarray(norm_cfg["q99"], dtype=np.float32)
    with open(SCENE_PKL, "rb") as f:
        scenes = pickle.load(f)
    by_token = {scene["token"]: scene for scene in scenes}

    boa_id = tokenizer.encode(tokenizer.boa_token)[0]
    eoa_id = tokenizer.encode(tokenizer.eoa_token)[0]
    LATENT_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"shard={SHARD_ID}/{NUM_SHARDS} scenes={len(files)} batch={BATCH_SIZE} "
        f"layers={LAYER_SPECS} output={LATENT_ROOT}", flush=True,
    )

    for file_i, input_path in enumerate(files):
        split = input_path.parent.name
        output_dir = LATENT_ROOT / split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.name
        if output_path.exists():
            print(f"latent {file_i:03d} {input_path.stem} status=exists", flush=True)
            continue

        with np.load(input_path, allow_pickle=False) as data:
            token = str(data["scene_token"])
            rel_raw = data["relative_trajectories"].astype(np.float32)
        scene = by_token[token]
        vq = np.load(resolve_vq_path(scene["image"][CUR_IDX])).reshape(1, 18, 32)
        context = (
            tokenizer.bos_token
            + scene["text"][CUR_IDX]
            + format_video_prompt(tokenizer, vq)
        )
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]

        rel_norm = 2.0 * (rel_raw - low) / (high - low) - 1.0
        rel_norm = np.clip(rel_norm, -1.0, 1.0)
        sequences = []
        eoa_positions = []
        last_vocab_idx = tokenizer.pad_token_id - 1
        for action in rel_norm:
            ids = action_tokenizer(torch.tensor(action, dtype=torch.float32))[0]
            action_ids = [boa_id] + [last_vocab_idx - int(x) for x in ids] + [eoa_id]
            sequence = context_ids + action_ids
            sequences.append(sequence)
            eoa_positions.append(len(sequence) - 1)

        latent_chunks = []
        resolved_layers = None
        with torch.inference_mode():
            for start in range(0, len(sequences), BATCH_SIZE):
                seq_batch = sequences[start:start + BATCH_SIZE]
                pos_batch = eoa_positions[start:start + BATCH_SIZE]
                max_len = max(map(len, seq_batch))
                input_ids = torch.full(
                    (len(seq_batch), max_len), tokenizer.pad_token_id,
                    dtype=torch.long, device=DEVICE,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row, sequence in enumerate(seq_batch):
                    input_ids[row, :len(sequence)] = torch.tensor(sequence, device=DEVICE)
                    attention_mask[row, :len(sequence)] = 1
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states
                if resolved_layers is None:
                    resolved_layers = [
                        spec if spec >= 0 else len(hidden_states) + spec
                        for spec in LAYER_SPECS
                    ]
                    if any(x < 0 or x >= len(hidden_states) for x in resolved_layers):
                        raise ValueError(
                            f"layers {LAYER_SPECS} invalid for {len(hidden_states)} hidden states"
                        )
                rows = torch.arange(len(seq_batch), device=DEVICE)
                positions = torch.tensor(pos_batch, device=DEVICE)
                selected = torch.stack(
                    [hidden_states[layer][rows, positions] for layer in resolved_layers], dim=1
                )
                latent_chunks.append(selected.to(torch.float16).cpu().numpy())

        latents = np.concatenate(latent_chunks, axis=0)
        tmp_path = output_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            np.savez_compressed(
                f,
                schema_version=np.asarray(1, dtype=np.int32),
                scene_token=np.asarray(token),
                split=np.asarray(split),
                layer_specs=np.asarray(LAYER_SPECS, dtype=np.int32),
                resolved_layers=np.asarray(resolved_layers, dtype=np.int32),
                eoa_latents=latents,
            )
        os.replace(tmp_path, output_path)
        print(
            f"latent {file_i:03d} scene={token} split={split} shape={latents.shape} "
            f"path={output_path}", flush=True,
        )


if __name__ == "__main__":
    main()
