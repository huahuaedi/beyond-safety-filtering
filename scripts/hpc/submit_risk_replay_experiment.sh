#!/bin/bash
set -euo pipefail
: "${DRIVEVLA_REPO_ROOT:?}" "${DRIVEVLA_ASSET_ROOT:?}" "${BASELINE_RUN:?}" "${BASELINE_REPORT_JOB:?}" "${SOURCE_MANIFEST:?}"
export DRIVEVLA_REPO_ROOT DRIVEVLA_ASSET_ROOT BASELINE_RUN SOURCE_MANIFEST
cd "$DRIVEVLA_REPO_ROOT"
STORE="$(readlink -f "$DRIVEVLA_ASSET_ROOT/pretrained_models")/DriveVLA-W0/posttrain_runs"
export REPLAY_RUN="$STORE/risk_replay_$(date +%Y%m%d_%H%M%S)"
mkdir "$REPLAY_RUN"
mkdir "$REPLAY_RUN/metrics"
cp tools/rl_prep/rl_stage1_prototype.py tools/rl_prep/risk_replay_experiment.py scripts/hpc/run_risk_replay_arm.sh "$REPLAY_RUN/"
submit() { sbatch --parsable --account=xinxia0 --nodes=1 --ntasks=1 --kill-on-invalid-dep=yes --output="$REPLAY_RUN/%x-%A_%a.log" --error="$REPLAY_RUN/%x-%A_%a.err" "$@"; }
record() { printf '%s=%s\n' "$1" "$2" | tee -a "$REPLAY_RUN/jobs.txt"; }
prep=$(submit --job-name=rr_prepare --partition=standard --cpus-per-task=2 --mem=8g --time=00:10:00 scripts/hpc/run_risk_replay_arm.sh prepare);prep=${prep%%;*};record prepare "$prep"
GPU=(--partition=gpu-rtx6000 --gres=gpu:rtx_pro_6000_blackwell:1)
smoke=$(submit --job-name=rr_smoke "${GPU[@]}" --cpus-per-task=8 --mem=64g --time=02:00:00 --array=0-1%2 --dependency="afterok:$prep:$BASELINE_REPORT_JOB" scripts/hpc/run_risk_replay_arm.sh smoke);smoke=${smoke%%;*};record smoke "$smoke"
train=$(submit --job-name=rr_train "${GPU[@]}" --cpus-per-task=8 --mem=64g --time=12:00:00 --array=0-5%5 --dependency="afterok:$smoke" scripts/hpc/run_risk_replay_arm.sh train);train=${train%%;*};record train "$train"
eval=$(submit --job-name=rr_eval "${GPU[@]}" --cpus-per-task=16 --mem=96g --time=04:00:00 --array=0-5%5 --dependency="afterok:$train" scripts/hpc/run_risk_replay_arm.sh eval);eval=${eval%%;*};record eval "$eval"
summary=$(submit --job-name=rr_summary --partition=standard --cpus-per-task=2 --mem=4g --time=00:10:00 --dependency="afterok:$eval" scripts/hpc/run_risk_replay_arm.sh summary);summary=${summary%%;*};record summary "$summary"
printf 'run=%s\n' "$REPLAY_RUN" | tee -a "$REPLAY_RUN/jobs.txt"
squeue -u "$USER"
