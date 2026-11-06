#!/bin/bash
# smoke first, then submit arms only after reviewing smoke output.
set -euo pipefail
stage="${1:?smoke|arms}"; run="${2:?existing RUN_DIR}"; pairs="${3:?random pairs.jsonl}"
root="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
cd "$root"
[[ "$stage" = smoke || "$stage" = arms ]] || exit 2
[[ -f "$run/manifest_v2.json" && -f "$pairs" ]] || { echo 'Missing manifest or pairs'; exit 2; }
export SCALED_RUN_DIR="$run" PAIRS_RANDOM="$pairs" SCALED_MODE_NAME=randompref
if [[ "$stage" = smoke ]]; then
 export SCALED_NAME=smoke_randompref SCALED_SEED=20 SCALED_UPDATES=2 SCALED_EVAL_SCENES=2
 [[ ! -e "$run/$SCALED_NAME" ]] || { echo 'Output exists'; exit 2; }
 sbatch --array=0-0 --time=01:30:00 scripts/hpc/submit_pair_pref_train.slurm
else
 [[ -f "$run/smoke_randompref/summary.json" ]] || { echo 'Missing completed smoke summary'; exit 2; }
 for seed in 20 21 22 23; do
  [[ ! -e "$run/scaled_v1_randompref_seed${seed}" ]] || { echo "Existing seed $seed"; exit 2; }
 done
 for seed in 20 21 22 23; do
  export SCALED_SEED="$seed" SCALED_NAME="scaled_v1_randompref_seed${seed}" SCALED_UPDATES=800 SCALED_EVAL_SCENES=600
  sbatch --array=0-0 scripts/hpc/submit_pair_pref_train.slurm
 done
fi
