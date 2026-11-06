"""
One-shot extraction: load the full 47GB formatted_pdm_score_8192.npy dict
exactly once, keep only entries for a small random sample of trainval scene
tokens, and dump that small subset to a new pickle so later runs of
stage0_5_worldmodel_fidelity.py never need to touch the 47GB file again.

(All 103288 scenes in formatted_pdm_score_8192.npy match the TRAINVAL split,
keyed by scene["token"] — confirmed by cross-checking sizes and a sample key
— not the test split we originally downloaded. This is presumably a training
artifact for the anchor/Hydra-MDP-style distillation head.)
"""
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

_PATH_PREFIXES = (
    "/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/",
    "/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/",
    "/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/",
)


def resolve_vq_path(raw_path: str, vq_root: Path) -> str:
    for old in _PATH_PREFIXES:
        if raw_path.startswith(old):
            return raw_path.replace(old, str(vq_root.parent) + "/")
    return raw_path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORE_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/formatted_pdm_score_8192.npy"
TRAINVAL_PKL = REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
OUT_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/trainval_scene_anchor_scores.pkl"
# train_vp_codes.zip only ships VQ codes for a subset of log segments (1192
# folders), not all 103288 trainval scenes — only sample scene tokens whose
# current-frame VQ file actually exists locally.
VQ_DIR = REPO_ROOT / "data/navsim/processed_data/data/navsim/processed_data/trainval_vq_codes_256_144"
CUR_IDX = 3
N_SAMPLE_SCENES = 30


def main():
    with open(TRAINVAL_PKL, "rb") as f:
        trainval_data = pickle.load(f)

    downloadable = []
    for scene in trainval_data:
        cur_path = resolve_vq_path(scene["image"][CUR_IDX], VQ_DIR)
        horizon_path = resolve_vq_path(scene["image"][CUR_IDX + 8], VQ_DIR)
        if os.path.exists(cur_path) and os.path.exists(horizon_path):
            downloadable.append(scene["token"])
    print(f"{len(downloadable)} / {len(trainval_data)} trainval scenes have both current+horizon VQ files available locally", file=sys.stderr, flush=True)

    random.seed(0)
    sample_tokens = set(random.sample(downloadable, min(N_SAMPLE_SCENES, len(downloadable))))
    print(f"Sampled {len(sample_tokens)} trainval scene tokens", file=sys.stderr, flush=True)
    test_tokens = sample_tokens

    print("Loading full anchor score dict (this is the expensive 47GB step)...", file=sys.stderr, flush=True)
    full = np.load(SCORE_PATH, allow_pickle=True).item()
    print(f"Loaded {len(full)} total entries", file=sys.stderr, flush=True)

    matched = {k: v for k, v in full.items() if k in test_tokens}
    print(f"Matched {len(matched)} entries against test scenes", file=sys.stderr, flush=True)
    if matched:
        print("sample matched key:", next(iter(matched)), file=sys.stderr, flush=True)
    else:
        print("sample full-dict key:", next(iter(full)), file=sys.stderr, flush=True)
        print("sample test token:", next(iter(test_tokens)), file=sys.stderr, flush=True)

    del full

    with open(OUT_PATH, "wb") as f:
        pickle.dump(matched, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
