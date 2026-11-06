"""
Runs inside the `drivevla` conda env on HPC. Builds a checkpoint-selection
validation subset -- disjoint from the 286-scene trainval_rl_eval held-out
set used for all final reported scores -- from world_action_full_v1's own
val split (751 scenes, itself disjoint from the RL training pool).

Mirrors build_eval_subset_pkl.py exactly (same subset-pkl/token-yaml schema,
consumed the same way by inference_action_navsim_flow_matching_vava.py's
--test_data_pkl/--token_yaml), except:
  - token source: world_action_full_v1/val/*.npz filenames, not the cached
    eval-token JSON.
  - a fixed-seed random subsample of those 751 (default 150) is used, not
    all of them -- full-751 real-NAVSIM-simulator eval per checkpoint is a
    lot of compute for a selection pass whose only job is to rank a
    training run's own checkpoints against each other, not to produce the
    number that gets reported.

Verify before trusting the output: this subset must have zero overlap with
trainval_rl_eval_tokens_cached.json (the 286 scenes everything else in this
investigation reports scores against) -- checked and asserted below, not
just assumed.
"""
import json
import pickle
import random
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_PKL = REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
EVAL_TOKENS_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/trainval_rl_eval_tokens_cached.json"
WORLD_ACTION_VAL_DIR = REPO_ROOT / "navsim_data/exp/world_action_full_v1/val"
OUT_PKL = REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_checkpoint_selection_val_pre_1s.pkl"
OUT_YAML = REPO_ROOT / "data/navsim/processed_data/scene_files/scene_filter/checkpoint_selection_val.yaml"
SUBSAMPLE_SIZE = 150
SUBSAMPLE_SEED = 0


def main():
    val_tokens = sorted(p.stem for p in WORLD_ACTION_VAL_DIR.glob("*.npz"))
    print(f"{len(val_tokens)} world_action_full_v1 val scenes", flush=True)

    eval_tokens = set(json.loads(EVAL_TOKENS_PATH.read_text()))
    overlap = set(val_tokens) & eval_tokens
    assert not overlap, (
        f"{len(overlap)} tokens overlap with the 286-scene reported eval "
        f"set -- refusing to build a leaky validation subset"
    )
    print(f"confirmed zero overlap with the {len(eval_tokens)}-scene reported eval set", flush=True)

    rng = random.Random(SUBSAMPLE_SEED)
    subsample_tokens = set(rng.sample(val_tokens, min(SUBSAMPLE_SIZE, len(val_tokens))))
    print(f"subsampled {len(subsample_tokens)} / {len(val_tokens)} val scenes "
          f"(seed={SUBSAMPLE_SEED}) for checkpoint selection", flush=True)

    with open(FULL_PKL, "rb") as f:
        full_data = pickle.load(f)

    subset = [entry for entry in full_data if entry["token"] in subsample_tokens]
    print(f"{len(subset)} / {len(subsample_tokens)} subsampled tokens found in trainval pkl", flush=True)
    missing = subsample_tokens - {entry["token"] for entry in subset}
    if missing:
        print(f"WARNING: {len(missing)} subsampled tokens not found in {FULL_PKL} -- "
              f"they will simply be absent from the output subset, not an error", flush=True)

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(subset, f)
    print(f"Wrote subset pkl to {OUT_PKL}", flush=True)

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump({"tokens": [entry["token"] for entry in subset]}, f)
    print(f"Wrote token yaml to {OUT_YAML}", flush=True)


if __name__ == "__main__":
    main()
