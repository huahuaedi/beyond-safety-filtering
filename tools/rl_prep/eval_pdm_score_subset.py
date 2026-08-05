"""
Runs inside the `navsim` conda env. Scores a directory of per-token predicted-
trajectory JSONs (produced by inference/vla/inference_action_navsim_flow_matching_vava.py,
Stage A) against a metric_cache, restricted to a fixed token subset — reuses
run_pdm_score.py's own scoring logic directly (calling its hydra-undecorated
function via `.__wrapped__`) instead of duplicating it, so this stays in sync
with the "real" eval pipeline used to originally report the 87.2 PDMS number.

Restricting to a subset via scene_filter.tokens is the same mechanism
tools/rl_prep/cache_trainval_rl_scenes.py already uses for the metric-caching
step (navsim.common.dataclasses.SceneFilter.tokens).

Usage:
    python eval_pdm_score_subset.py \
        --experiment-path /path/to/stage_a_json_output_dir \
        --experiment-name before_rl_trainval_eval \
        --metric-cache-path $NAVSIM_EXP_ROOT/metric_cache_trainval_rl_eval \
        --subset-tokens-json data/navsim/processed_data/anchor/trainval_rl_eval_tokens_cached.json \
        --train-test-split trainval
"""
import argparse
import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference/navsim/navsim"))

from navsim.planning.script import run_pdm_score as run_pdm_score_module  # noqa: E402

CONFIG_DIR = REPO_ROOT / "inference/navsim/navsim/navsim/planning/script/config/pdm_scoring"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-path", required=True, help="Stage A output_dir (per-token prediction JSONs)")
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--metric-cache-path", required=True)
    ap.add_argument("--subset-tokens-json", required=True)
    ap.add_argument("--train-test-split", default="trainval", help="hydra train_test_split group to compose")
    ap.add_argument("--anchor-based", default="false")
    args = ap.parse_args()

    with open(args.subset_tokens_json) as f:
        subset_tokens = json.load(f)
    print(f"Scoring against {len(subset_tokens)} held-out tokens", flush=True)

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="default_run_pdm_score",
            overrides=[
                f"train_test_split={args.train_test_split}",
                "agent=emu_vla_agent",
                f"agent.experiment_path={args.experiment_path}",
                f"agent.anchor_based={args.anchor_based}",
                f"experiment_name={args.experiment_name}",
                f"metric_cache_path={args.metric_cache_path}",
            ],
        )
    cfg.train_test_split.scene_filter.tokens = subset_tokens

    # Bypassing the @hydra.main CLI decorator (calling the raw function via
    # __wrapped__) means Hydra's own run-dir auto-creation never fires —
    # main()'s body writes straight to cfg.output_dir, so make sure it exists.
    os.makedirs(cfg.output_dir, exist_ok=True)

    run_pdm_score_module.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
