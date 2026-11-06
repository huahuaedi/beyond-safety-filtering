#!/bin/bash
# Larger KL budgets for the tilt: eps 1.0 and 2.0, seeds 20 and 21, same recipe as the v2 study arms (no preservation).
# The study ran eps 0 and 0.3 only, whose promised gain in pass probability is small (about 4 percentage points); best-of-16 corresponds to
# a budget near ln 16 = 2.8 nats. Where a budget exceeds what the pool can reach the weights concentrate on the passing candidates.
# Usage: bash scripts/hpc/submit_awfm_highbudget.sh <RUN_DIR with manifest_v2.json, the SFT baseline and the v2 smoke>
set -euo pipefail
RUN_DIR="${1:?existing directory with manifest_v2.json and baseline}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
cd "$ROOT"
export SCALED_RUN_DIR="$RUN_DIR"
export AWFM_OUTPUT_PREFIX=scaled_awfm_v2_
export AWFM_REPORT_STAGE=highbudget
[ -f "$RUN_DIR/manifest_v2.json" ] || { echo 'missing manifest'; exit 1; }
PY="$HOME/.conda/envs/drivevla/bin/python"
export AWFM_BASE
AWFM_BASE=$("$PY" tools/rl_prep/check_awfm_inputs.py "$RUN_DIR" \
 --metric-cache-dir "$ROOT/navsim_data/exp/metric_cache_trainval_rl" \
 --eval-scenes "${SCALED_EVAL_SCENES:-600}" --base-only)
unset AWFM_EPS AWFM_TAG AWFM_PRESERVE SCALED_SEED SCALED_NAME AWFM_SMOKE
# the untouched-parameter smoke (preserve none) must already have passed in this run directory
"$PY" tools/rl_prep/check_awfm_run.py "$RUN_DIR/smoke_awfm_v2_none" --preserve none --expected-scenes 2 --expected-updates 2 >&2
for tag in e1p0 e2p0; do for seed in 20 21; do
  [ ! -e "$RUN_DIR/${AWFM_OUTPUT_PREFIX}awfm${tag}_seed${seed}" ] || { echo "refusing existing ${AWFM_OUTPUT_PREFIX}awfm${tag}_seed${seed}"; exit 1; }
done; done
JOBS=""
for spec in "1.0 e1p0" "2.0 e2p0"; do
  set -- $spec
  for seed in 20 21; do
    name="${AWFM_OUTPUT_PREFIX}awfm${2}_seed${seed}"
    J=$(sbatch --parsable --array=0-0 \
        --export=ALL,AWFM_EPS="$1",AWFM_TAG="$2",AWFM_PRESERVE=none,SCALED_SEED="$seed",SCALED_NAME="$name" scripts/hpc/submit_awfm_train.slurm)
    echo "$name -> job $J"
    JOBS="${JOBS:+$JOBS:}$J"
  done
done
export AWFM_REQUIRED="awfme0p0:20,21 awfme1p0:20,21 awfme2p0:20,21"
S=$(sbatch --parsable --dependency="afterok:$JOBS" scripts/hpc/submit_awfm_summary.slurm)
echo "summary=$S   report=$RUN_DIR/awfm_v2_highbudget_comparison.json"
