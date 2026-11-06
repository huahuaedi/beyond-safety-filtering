#!/bin/bash
# Complete Action Expert RL post-training, validation selection, final evaluation.
set -euo pipefail
: "${DRIVEVLA_REPO_ROOT:?set code checkout}"
: "${DRIVEVLA_ASSET_ROOT:?set original repository with pretrained_models/data/navsim_data}"
export DRIVEVLA_REPO_ROOT DRIVEVLA_ASSET_ROOT
cd "$DRIVEVLA_REPO_ROOT"
STORE="$(readlink -f "$DRIVEVLA_ASSET_ROOT/pretrained_models")/DriveVLA-W0/posttrain_runs"
export POSTTRAIN_RUN="$STORE/posttrain_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$STORE"
mkdir "$POSTTRAIN_RUN"
mkdir "$POSTTRAIN_RUN/metrics"
cp tools/rl_prep/rl_stage1_prototype.py tools/rl_prep/policy_optimization.py scripts/hpc/run_posttrain_stage.sh "$POSTTRAIN_RUN/"
git rev-parse HEAD > "$POSTTRAIN_RUN/base_commit.txt"
git diff --binary HEAD > "$POSTTRAIN_RUN/source_changes.patch"
submit() {
 sbatch --parsable --account=xinxia0 --nodes=1 --ntasks=1 --kill-on-invalid-dep=yes \
 --output="$POSTTRAIN_RUN/%x-%A_%a.log" --error="$POSTTRAIN_RUN/%x-%A_%a.err" "$@"
}
record() { printf '%s=%s\n' "$1" "$2" | tee -a "$POSTTRAIN_RUN/jobs.txt"; }
prep=$(submit --job-name=pt_prepare --partition=standard --cpus-per-task=2 --mem=16g --time=00:15:00 scripts/hpc/run_posttrain_stage.sh prepare); prep=${prep%%;*}; record prepare "$prep"
GPU=(--partition=gpu-rtx6000 --gres=gpu:rtx_pro_6000_blackwell:1)
smoke=$(submit --job-name=pt_smoke "${GPU[@]}" --cpus-per-task=8 --mem=64g --time=02:00:00 --dependency="afterok:$prep" scripts/hpc/run_posttrain_stage.sh smoke); smoke=${smoke%%;*}; record smoke "$smoke"
base=$(submit --job-name=pt_val_baseline "${GPU[@]}" --cpus-per-task=16 --mem=96g --time=04:00:00 --dependency="afterok:$prep" scripts/hpc/run_posttrain_stage.sh val_baseline); base=${base%%;*}; record baseline "$base"
train=$(submit --job-name=pt_train "${GPU[@]}" --cpus-per-task=8 --mem=64g --time=12:00:00 --dependency="afterok:$smoke" scripts/hpc/run_posttrain_stage.sh train); train=${train%%;*}; record train "$train"
val=$(submit --job-name=pt_val "${GPU[@]}" --cpus-per-task=16 --mem=96g --time=04:00:00 --array=0-3%4 --dependency="afterok:$train" scripts/hpc/run_posttrain_stage.sh val); val=${val%%;*}; record val "$val"
select=$(submit --job-name=pt_select --partition=standard --cpus-per-task=2 --mem=4g --time=00:10:00 --dependency="afterok:$val:$base" scripts/hpc/run_posttrain_stage.sh select); select=${select%%;*}; record select "$select"
test=$(submit --job-name=pt_test "${GPU[@]}" --cpus-per-task=16 --mem=96g --time=04:00:00 --array=0-1%2 --dependency="afterok:$select" scripts/hpc/run_posttrain_stage.sh test); test=${test%%;*}; record test "$test"
report=$(submit --job-name=pt_report --partition=standard --cpus-per-task=2 --mem=4g --time=00:10:00 --dependency="afterok:$test" scripts/hpc/run_posttrain_stage.sh final); report=${report%%;*}; record report "$report"
printf 'run=%s\n' "$POSTTRAIN_RUN" | tee -a "$POSTTRAIN_RUN/jobs.txt"
squeue -u "$USER"
