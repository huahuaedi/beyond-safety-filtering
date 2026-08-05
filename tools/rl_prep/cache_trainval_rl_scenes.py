"""
Runs inside the `navsim` conda env. Builds a NAVSIM metric_cache for a batch
of candidate trainval scene tokens (any tokens we have local VQ data for —
see tools/rl_prep/rl_stage1_prototype.py's REWARD_SOURCE=navsim path, which
no longer needs the anchor-score pickle at all).

Important: candidate tokens come from DriveVLA-W0's own data pipeline, which
samples frames on a different grid than NAVSIM's official scene extraction
(fixed num_history_frames/num_future_frames/frame_interval). Only a fraction
of candidates will actually land on a valid NAVSIM scene anchor — this script
oversamples candidates and reports which ones actually resolved, writing that
*intersected* set to trainval_rl_scene_tokens_cached.json for the RL script
to use as its scene pool.

Usage (env vars matching the rest of the navsim setup):
    export OPENSCENE_DATA_ROOT=... NAVSIM_EXP_ROOT=... NUPLAN_MAPS_ROOT=... \
           NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_DEVKIT_ROOT=...
    python cache_trainval_rl_scenes.py                 # training pool (default)
    RL_CACHE_TARGET=eval python cache_trainval_rl_scenes.py  # held-out eval pool
      (a *disjoint* token set, see tools/rl_prep/sample_trainval_scene_pool.py —
      used for before/after PDMS comparison, never sampled during RL training)
"""
import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_DIR = REPO_ROOT / "data/navsim/processed_data/anchor"
CONFIG_DIR = REPO_ROOT / "inference/navsim/navsim/navsim/planning/script/config/metric_caching"

# Oversampled candidate pool (any trainval token with local VQ data — see
# extract logic in extract_relevant_anchor_scores.py, reused ad hoc when this
# pool was generated). Plain JSON, not the (numpy 2.x, drivevla-env-only)
# anchor-score pickle.
CACHE_TARGET = os.environ.get("RL_CACHE_TARGET", "train")
if CACHE_TARGET == "eval":
    CANDIDATE_TOKENS_PATH = ANCHOR_DIR / "trainval_rl_eval_tokens_pool.json"
    FOUND_TOKENS_PATH = ANCHOR_DIR / "trainval_rl_eval_tokens_cached.json"
    CACHE_SUBDIR = "metric_cache_trainval_rl_eval"
else:
    CANDIDATE_TOKENS_PATH = ANCHOR_DIR / "trainval_rl_scene_tokens_pool.json"
    FOUND_TOKENS_PATH = ANCHOR_DIR / "trainval_rl_scene_tokens_cached.json"
    CACHE_SUBDIR = "metric_cache_trainval_rl"

sys.path.insert(0, str(REPO_ROOT / "inference/navsim/navsim"))

from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.planning.metric_caching.caching import cache_data  # noqa: E402
from navsim.planning.script.builders.worker_pool_builder import build_worker  # noqa: E402


def main():
    with open(CANDIDATE_TOKENS_PATH) as f:
        candidate_tokens = json.load(f)
    print(f"{len(candidate_tokens)} candidate tokens (oversampled)", flush=True)

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="default_metric_caching",
            overrides=[
                "train_test_split=trainval",
                f"cache.cache_path={os.environ['NAVSIM_EXP_ROOT']}/{CACHE_SUBDIR}",
            ],
        )
    cfg.train_test_split.scene_filter.tokens = candidate_tokens

    # Check which candidates actually resolve to a valid NAVSIM scene anchor
    # *before* spending time caching all of them.
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    found_tokens = list(scene_loader.tokens)
    print(f"{len(found_tokens)} / {len(candidate_tokens)} candidates are valid NAVSIM scene anchors", flush=True)
    with open(FOUND_TOKENS_PATH, "w") as f:
        json.dump(found_tokens, f)

    worker = build_worker(cfg)
    cache_data(cfg=cfg, worker=worker)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
