#!/bin/bash
# Re-collect 16 candidates per scene on the frozen navtest cohort and run geometry + the frozen selector on them.
# Usage:  bash scripts/hpc/run_navtest_rerun.sh <seed> [policy-checkpoint-dir] [label]
#   seed:   sampling seed of the candidates (the reference probes do not depend on it)
#   policy: optional action-expert checkpoint to sample from (default: the SFT policy)
#   label:  optional name part of the output directory (default: seed<seed>)
set -euo pipefail
SEED="${1:?seed, e.g. 1}"
POLICY="${2:-}"
LABEL="${3:-seed$SEED}"
cd "$HOME/DriveVLA-Safety-RL-offline-20260907"
export DRIVEVLA_REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
export OFFLINE_REPO_ROOT="$HOME/DriveVLA-Safety-RL-offline-20260907"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
export NT_COHORT="$ASSET/pretrained_models/DriveVLA-W0/posttrain_runs/rl_pareto_navtest_v2_20261005_133703_37354"
export NT_OUT="$ASSET/navsim_data/exp/navtest_selector_e2e_$(date +%Y%m%d_%H%M%S)_$LABEL"
export NT_STEPS=100000 NT_GROUP=16 RL_SEED="$SEED"
BASE_CSV=""
if [ -n "$POLICY" ]; then
  export NT_POLICY="$POLICY"
  # the baseline of a policy is its own independent evaluation on the same cohort (label = arm name of orig_five_navtest)
  BASE_CSV=$(ls "$ASSET"/pretrained_models/DriveVLA-W0/posttrain_runs/orig_five_navtest_*/metrics/*_"$LABEL"/*/*.csv 2>/dev/null | grep -v smoke | tail -1 || true)
  [ -n "$BASE_CSV" ] || { echo "no independent evaluation csv for arm '$LABEL'; pass the arm name (e.g. sel_seed20_step000025) as 3rd argument" >&2; exit 6; }
  echo "baseline csv: $BASE_CSV"
fi
test -f "$DRIVEVLA_REPO_ROOT/tools/rl_prep/rl_stage1_prototype.py"
mkdir "$NT_OUT"
J=$(sbatch --parsable --array='0-3%4' scripts/hpc/submit_navtest_candidates.slurm)
echo "collection job=$J seed=$SEED policy=${POLICY:-SFT} out=$NT_OUT"
CHAIN_BASE_CSV="$BASE_CSV" CHAIN_OUT="$NT_OUT" CHAIN_JOB="$J" bash scripts/hpc/run_navtest_selector_e2e.sh chain
echo "result will be: $NT_OUT/selector_navtest.json"
