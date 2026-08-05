"""
Runs inside the `drivevla` conda env, AFTER
    RL_CACHE_TARGET=eval python tools/rl_prep/cache_trainval_rl_scenes.py
has produced data/navsim/processed_data/anchor/trainval_rl_eval_tokens_cached.json
(the final, NAVSIM-scene-anchor-validated held-out eval token set — disjoint
from the RL training pool, see sample_trainval_scene_pool.py).

Filters the full trainval metadata pkl down to just those tokens and writes:
  - a subset pkl, same schema as navsim_emu_vla_256_144_trainval_pre_1s.pkl
    (consumed by inference_action_navsim_flow_matching_vava.py's --test_data_pkl)
  - a matching token YAML, row-order-aligned with the subset pkl (consumed by
    that same script's --token_yaml, used only for naming output JSON files by
    scene token — see inference/vla/inference_action_navsim_flow_matching_vava.py:184)

This sidesteps needing the (missing, not reproducible) official navtest.yaml —
trainval pkl entries carry their own "token" field directly, unlike the test
split pkl, so no separate ordered-mapping file is needed here.
"""
import json
import pickle
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_PKL = REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
EVAL_TOKENS_PATH = REPO_ROOT / "data/navsim/processed_data/anchor/trainval_rl_eval_tokens_cached.json"
OUT_PKL = REPO_ROOT / "pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_rl_eval_pre_1s.pkl"
OUT_YAML = REPO_ROOT / "data/navsim/processed_data/scene_files/scene_filter/trainval_rl_eval.yaml"


def main():
    with open(EVAL_TOKENS_PATH) as f:
        eval_tokens = set(json.load(f))
    print(f"{len(eval_tokens)} held-out eval tokens", flush=True)

    with open(FULL_PKL, "rb") as f:
        full_data = pickle.load(f)

    subset = [entry for entry in full_data if entry["token"] in eval_tokens]
    print(f"{len(subset)} / {len(eval_tokens)} eval tokens found in trainval pkl", flush=True)

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
