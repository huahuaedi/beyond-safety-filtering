#!/bin/bash
# Submit one NAVSIM v2 re-scoring job per evaluated policy label and traffic mode.
# Usage: bash scripts/hpc/run_v2_rescore_all.sh <CACHE_JOB_ID|none> <TRAFFIC: non_reactive|reactive|both> LABEL [LABEL ...]
# Only labels whose merged shard directories exist are submitted; a label with no npz is skipped and reported.
# Example: bash scripts/hpc/run_v2_rescore_all.sh 65354906 both sft nearestpref_s20 pdmspref_s20 marginpref_s20
set -euo pipefail
DEP="${1:?cache job id or none}"; MODE="${2:?non_reactive|reactive|both}"; shift 2
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
OUT=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data/navsim_data/exp/official_navtest_trained_eval
[ "$MODE" = both ] && MODES="non_reactive reactive" || MODES="$MODE"
DEPARG=(); [ "$DEP" != none ] && DEPARG=(--dependency="afterok:$DEP")
cd "$ROOT"
for L in "$@"; do
  if ! ls -d "$OUT/${L}__shard"* >/dev/null 2>&1; then echo "SKIP $L: no $OUT/${L}__shard*"; continue; fi
  for T in $MODES; do
    J=$(sbatch --parsable "${DEPARG[@]}" --job-name="v2_${T:0:3}_$L" --export=ALL,LABEL="$L",TRAFFIC="$T" scripts/hpc/submit_v2_rescore.slurm)
    echo "$L $T -> $J"
  done
done
