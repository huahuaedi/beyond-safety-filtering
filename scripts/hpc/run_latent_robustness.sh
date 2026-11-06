#!/bin/bash
# Evaluation-noise floor and latent robustness on the full official navtest (12,146 scenes, 12 shards per entry).
# The default evaluation uses one fixed per-scene initial latent (tag offline_eval_v1). Here the SFT policy is scored under two other
# tags (noise floor of the evaluator) and three trained arms under tag L2 (do their paired gains depend on that one latent draw?).
# Usage: bash scripts/hpc/run_latent_robustness.sh      Merge afterwards (one label at a time):
#   python merge_policy_eval_shards.py $X/official_navtest_trained_eval <label> $M/<label> --expect-scenes 12146
# and compare sft_L2 vs sft (noise floor) and e.g. nearestpref_s20_L2 vs sft_L2 (same tag) with paired_scores_by_log.py.
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
RUN="${1:-$ROOT/output/offline_scaled_20261008_150131}"
X=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data/navsim_data/exp
OUT=$X/official_navtest_trained_eval
cd "$ROOT"
submit() {   # tag, list lines...
  local tag="$1"; shift
  local list="$OUT/arms_latent_${tag}.txt"
  [ -e "$list" ] && { echo "refusing existing $list"; exit 1; }
  printf '%s\n' "$@" > "$list"
  local n=$#
  J=$(EVAL_LIST="$list" EVAL_OUT="$OUT" EVAL_SHARDS=12 EVAL_LATENT_TAG="$tag" NT_COHORT="$X/official_navtest_full_cohort2" \
      sbatch --parsable --array=0-$((n*12-1))%5 scripts/hpc/submit_navtest_policy_eval.slurm)
  echo "tag $tag: $n entries -> job $J"
}
submit L2 "sft_L2|-" \
  "nearestpref_s20_L2|$RUN/scaled_v1_nearestpref_seed20/action_policy.pt" \
  "marginpref_s20_L2|$RUN/scaled_v1_marginpref_seed20/action_policy.pt" \
  "pdmspref_s20_L2|$RUN/scaled_v1_pdmspref_seed20/action_policy.pt"
submit L3 "sft_L3|-"
