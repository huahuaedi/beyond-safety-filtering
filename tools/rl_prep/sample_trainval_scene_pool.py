"""
Runs inside the `drivevla` conda env. Produces the oversampled candidate-token
pool consumed by tools/rl_prep/cache_trainval_rl_scenes.py (which runs in the
separate `navsim` env and can't load Emu3DrivingVAVADataset itself).

DriveVLA-W0's own frame sampling doesn't align 1:1 with NAVSIM's official
scene grid, so only a fraction of any candidate pool will actually resolve to
a valid NAVSIM scene anchor (locally: 18/300) — oversample here, let
cache_trainval_rl_scenes.py report which ones actually resolved.

Also carves out a second, disjoint candidate pool (RL_EVAL_POOL_SIZE, default
0 = skip) for held-out before/after PDMS comparison (see docs/rl_world_model_design.md
阶段 2) — these tokens are guaranteed to never appear in the training pool,
since both are sliced from a single fixed shuffle of the full token list.

random.seed(1) is fixed so this is reproducible across machines (e.g. to
regenerate the same pools on HPC instead of copying the local-only json over).

Set RL_USE_ALL_TRAIN_SCENES=1 to write every unique VQ-available trainval
token except the frozen held-out candidate pool.  In that mode the existing
eval-pool JSON is preserved by default (RL_FREEZE_EVAL_POOL=1), which prevents
an expanded training pool from silently swallowing the held-out evaluation
scenes used by earlier experiments.
"""
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference" / "vla"))
sys.path.insert(0, str(REPO_ROOT / "utils"))
sys.path.insert(0, str(REPO_ROOT / "reference" / "Emu3"))
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("VLA_ACTION_TOKENIZER", str(REPO_ROOT / "pretrained_models/fast"))
os.environ.setdefault("VLA_VLM_MODEL", str(REPO_ROOT / "pretrained_models/Emu3-Stage1"))
os.environ.setdefault("VLA_NORM_STATS", str(REPO_ROOT / "configs/normalizer_navsim_trainval/norm_stats.json"))
os.environ.setdefault(
    "VLA_PATH_REPLACEMENTS",
    "/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:"
    f"{REPO_ROOT}/data/navsim/processed_data/data/navsim/processed_data/;"
    "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:"
    f"{REPO_ROOT}/data/navsim/processed_data/data/navsim/processed_data/;"
    "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:"
    f"{REPO_ROOT}/data/navsim/processed_data/data/navsim/processed_data/",
)

from config import setup_paths_early  # noqa: E402
setup_paths_early()

from emu3.mllm import Emu3Tokenizer  # noqa: E402
from utils.train_pi0 import DataArguments  # noqa: E402
from datasets import Emu3DrivingVAVADataset  # noqa: E402

SCENE_PKL = REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
POOL_SIZE = int(os.environ.get("RL_SCENE_POOL_SIZE", 300))
EVAL_POOL_SIZE = int(os.environ.get("RL_EVAL_POOL_SIZE", 0))
USE_ALL_TRAIN_SCENES = os.environ.get("RL_USE_ALL_TRAIN_SCENES", "0") == "1"
FREEZE_EVAL_POOL = os.environ.get("RL_FREEZE_EVAL_POOL", "1") == "1"
OUT_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/trainval_rl_scene_tokens_pool.json"
EVAL_OUT_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/trainval_rl_eval_tokens_pool.json"


def main():
    tokenizer = Emu3Tokenizer.from_pretrained(
        os.environ["VLA_VLM_MODEL"], model_max_length=1400, padding_side="right", use_fast=False,
    )
    data_args = DataArguments(
        data_path=str(SCENE_PKL),
        actions=True, driving=True, use_previous_actions=True, actions_format="fast",
        action_tokenizer_path=os.environ["VLA_ACTION_TOKENIZER"],
        frames=1, action_frames=8, action_dim=3, cur_frame_idx=3, pre_action_frames=3,
        video_format=None, use_flip=False,
    )
    dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    tokens = [scene["token"] for scene in dataset.data]
    unique_tokens = list(dict.fromkeys(tokens))
    print(
        f"{len(tokens)} VQ-available trainval entries; "
        f"{len(unique_tokens)} unique scene tokens",
        flush=True,
    )

    random.seed(1)
    shuffled = unique_tokens.copy()
    random.shuffle(shuffled)

    existing_eval_pool = None
    if EVAL_OUT_PATH.exists() and FREEZE_EVAL_POOL:
        with open(EVAL_OUT_PATH) as f:
            existing_eval_pool = json.load(f)
        if len(existing_eval_pool) != len(set(existing_eval_pool)):
            raise ValueError(f"Frozen eval pool contains duplicate tokens: {EVAL_OUT_PATH}")
        unknown_eval = set(existing_eval_pool) - set(unique_tokens)
        if unknown_eval:
            raise ValueError(
                f"Frozen eval pool contains {len(unknown_eval)} tokens absent from the current trainval dataset"
            )
        print(
            f"Preserving frozen held-out candidate pool with {len(existing_eval_pool)} tokens: "
            f"{EVAL_OUT_PATH}",
            flush=True,
        )

    if USE_ALL_TRAIN_SCENES:
        if existing_eval_pool is None:
            if EVAL_POOL_SIZE <= 0:
                raise ValueError(
                    "RL_USE_ALL_TRAIN_SCENES=1 requires either an existing frozen eval-pool JSON "
                    "or RL_EVAL_POOL_SIZE > 0"
                )
            existing_eval_pool = shuffled[:EVAL_POOL_SIZE]
            with open(EVAL_OUT_PATH, "w") as f:
                json.dump(existing_eval_pool, f)
            print(
                f"Created and froze held-out candidate pool with {len(existing_eval_pool)} tokens: "
                f"{EVAL_OUT_PATH}",
                flush=True,
            )
        eval_tokens = set(existing_eval_pool)
        pool = [token for token in shuffled if token not in eval_tokens]
    else:
        pool = shuffled[:POOL_SIZE]

    overlap = set(pool) & set(existing_eval_pool or [])
    if overlap:
        raise RuntimeError(f"Training/eval candidate leakage detected: {len(overlap)} overlapping tokens")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(pool, f)
    mode = "all available minus held-out" if USE_ALL_TRAIN_SCENES else "sampled"
    print(f"Wrote {len(pool)} {mode} training candidate tokens to {OUT_PATH}", flush=True)

    if not USE_ALL_TRAIN_SCENES and EVAL_POOL_SIZE > 0 and existing_eval_pool is None:
        # Sliced from *after* the training pool in the same shuffle — disjoint
        # by construction, no overlap check needed.
        eval_pool = shuffled[POOL_SIZE:POOL_SIZE + EVAL_POOL_SIZE]
        with open(EVAL_OUT_PATH, "w") as f:
            json.dump(eval_pool, f)
        print(f"Wrote {len(eval_pool)} held-out eval candidate tokens to {EVAL_OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
