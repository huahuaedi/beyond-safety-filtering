#!/bin/bash
# PDMS-winner preference distillation, seeds 22 and 23 (same pre-registered settings as seeds 20, 21), then sharded official-navtest evaluation.
# Usage: bash scripts/hpc/run_pdms_seeds_navtest.sh [RUN_DIR]     RUN_DIR defaults to offline_scaled_20261008_150131.
# Afterwards, per arm:  python merge_policy_eval_shards.py $X/official_navtest_trained_eval pdmspref_s22 $M/pdmspref_s22 --expect-scenes 12146
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
RUN="${1:-$ROOT/output/offline_scaled_20261008_150131}"
X=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data/navsim_data/exp
cd "$ROOT"
export SCALED_RUN_DIR="$RUN"
export PAIRS_MARGIN="${PAIRS_MARGIN:-$ROOT/output/prefs_v3_margin/pairs.jsonl}"
export PAIRS_PDMS="${PAIRS_PDMS:-$ROOT/output/prefs_v3_pdms/pairs.jsonl}"
export PAIRS_NEAREST="${PAIRS_NEAREST:-$ROOT/output/prefs_v3_nearest/pairs.jsonl}"
for f in "$RUN/manifest_v2.json" "$PAIRS_PDMS"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
OUT=$X/official_navtest_trained_eval
LIST=$OUT/arms_pdms2223.txt
[ -e "$LIST" ] && { echo "refusing existing $LIST"; exit 1; }
for s in 22 23; do [ -e "$RUN/scaled_v1_pdmspref_seed${s}" ] && { echo "refusing existing $RUN/scaled_v1_pdmspref_seed${s}"; exit 1; }; done
JOBS=()
for s in 22 23; do
  J=$(sbatch --parsable --array=0-0 --export=ALL,SCALED_MODE_NAME=pdmspref,SCALED_SEED=$s scripts/hpc/submit_pair_pref_train.slurm)
  echo "pdmspref seed $s -> job $J"; JOBS+=("$J")
  echo "pdmspref_s${s}|$RUN/scaled_v1_pdmspref_seed${s}/action_policy.pt" >> "$LIST"
done
DEP="afterok:$(IFS=:; echo "${JOBS[*]}")"
export EVAL_LIST="$LIST" EVAL_OUT="$OUT" EVAL_SHARDS=12 NT_COHORT=$X/official_navtest_full_cohort2
J=$(sbatch --parsable --array=0-23%5 --dependency="$DEP" scripts/hpc/submit_navtest_policy_eval.slurm)
echo "eval array $J (after $DEP); output $OUT"
