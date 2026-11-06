#!/bin/bash
# Fresh v2 output names. study = eps0 / eps0.3 / global / targeted, two seeds each.
set -euo pipefail
STAGE="${1:?smoke|study|arms|all|preserve}"
RUN_DIR="${2:?existing directory with manifest_v2.json and baseline}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
cd "$ROOT"
export SCALED_RUN_DIR="$RUN_DIR"
export AWFM_OUTPUT_PREFIX=scaled_awfm_v2_
export AWFM_REPORT_STAGE="$STAGE"
[ -f "$RUN_DIR/manifest_v2.json" ] || { echo 'missing manifest'; exit 1; }
PY="$HOME/.conda/envs/drivevla/bin/python"
# Fail before submitting GPUs if the reused baseline/cohort/cache do not match.
export AWFM_BASE
AWFM_BASE=$("$PY" tools/rl_prep/check_awfm_inputs.py "$RUN_DIR" \
 --metric-cache-dir "$ROOT/navsim_data/exp/metric_cache_trainval_rl" \
 --eval-scenes "${SCALED_EVAL_SCENES:-600}" --base-only)
# Prevent stale single-arm overrides from silently making every array task identical.
unset AWFM_EPS AWFM_TAG AWFM_PRESERVE SCALED_SEED SCALED_NAME AWFM_SMOKE
smoke() {
  local name="smoke_awfm_v2_$1"
  [ ! -e "$RUN_DIR/$name" ] || { echo "refusing existing $name" >&2; return 1; }
  sbatch --parsable --array=0-0 --time=01:30:00 \
    --export=ALL,AWFM_EPS=0.3,AWFM_TAG=e0p3,AWFM_PRESERVE="$1",SCALED_SEED=20,SCALED_NAME="$name",SCALED_UPDATES=2,SCALED_EVAL_SCENES=2,AWFM_SMOKE=1 \
    scripts/hpc/submit_awfm_train.slurm
}
ensure_smoke() {
  local name="smoke_awfm_v2_$1"
  if [ -e "$RUN_DIR/$name" ]; then
    "$PY" tools/rl_prep/check_awfm_run.py "$RUN_DIR/$name" --preserve "$1" --expected-scenes 2 --expected-updates 2 >&2
  else
    local job
    job=$(smoke "$1")
    echo "smoke $1=$job" >&2
    DEP="${DEP:+$DEP:}$job"
  fi
}
case "$STAGE" in
  smoke) J=$(smoke targeted); echo "targeted smoke=$J"; exit 0 ;;
  study) TASKS='2-3,6-11%5'; TAGS='e0p0 e0p3 gloe0p3 tgte0p3'; NEED_TARGETED=1 ;;
  arms|all) TASKS='0-5%5'; TAGS='e0p1 e0p3 e1p0'; NEED_TARGETED=0 ;;
  preserve) TASKS='6-11%5'; TAGS='e0p0 gloe0p3 tgte0p3'; NEED_TARGETED=1 ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
export AWFM_REQUIRED=""
for tag in $TAGS; do
  AWFM_REQUIRED="${AWFM_REQUIRED} awfm${tag}:20,21"
  for seed in 20 21; do
    name="${AWFM_OUTPUT_PREFIX}awfm${tag}_seed${seed}"
    [ ! -e "$RUN_DIR/$name" ] || { echo "refusing existing $name"; exit 1; }
  done
done
DEP=''
ensure_smoke none
if [ "$NEED_TARGETED" = 1 ]; then ensure_smoke targeted; fi
DEPS=()
if [ -n "$DEP" ]; then DEPS=(--dependency="afterok:$DEP"); fi
J=$(sbatch --parsable "${DEPS[@]}" --array="$TASKS" scripts/hpc/submit_awfm_train.slurm)
echo "train=$J"
S=$(sbatch --parsable --dependency="afterok:$J" scripts/hpc/submit_awfm_summary.slurm)
echo "summary=$S"
echo "outputs=$RUN_DIR/${AWFM_OUTPUT_PREFIX}*"
echo "report=$RUN_DIR/awfm_v2_${STAGE}_comparison.json"
