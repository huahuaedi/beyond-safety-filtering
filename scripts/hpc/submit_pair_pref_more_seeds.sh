#!/bin/bash
# Two more training seeds (22, 23) for the margin and nearest-reference preference arms; same pre-registered settings.
# Usage: bash scripts/hpc/submit_pair_pref_more_seeds.sh [RUN_DIR]     prints the job ids (use them as EVAL_DEP for the navtest evaluation).
set -euo pipefail
RUN="${1:-$HOME/DriveVLA-Safety-RL-offline-20260907/output/offline_scaled_20261008_150131}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
cd "$ROOT"
export SCALED_RUN_DIR="$RUN"
export PAIRS_MARGIN="${PAIRS_MARGIN:-$ROOT/output/prefs_v3_margin/pairs.jsonl}"
export PAIRS_PDMS="${PAIRS_PDMS:-$ROOT/output/prefs_v3_pdms/pairs.jsonl}"
export PAIRS_NEAREST="${PAIRS_NEAREST:-$ROOT/output/prefs_v3_nearest/pairs.jsonl}"
for f in "$RUN/manifest_v2.json" "$PAIRS_MARGIN" "$PAIRS_NEAREST"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
JOBS=()
for a in marginpref nearestpref; do for s in 22 23; do
  [ -e "$RUN/scaled_v1_${a}_seed${s}" ] && { echo "refusing existing $RUN/scaled_v1_${a}_seed${s}"; exit 1; }
  J=$(sbatch --parsable --array=0-0 --export=ALL,SCALED_MODE_NAME=$a,SCALED_SEED=$s scripts/hpc/submit_pair_pref_train.slurm)
  echo "$a seed $s -> job $J"; JOBS+=("$J")
done; done
echo "EVAL_DEP=afterok:$(IFS=:; echo "${JOBS[*]}")"
