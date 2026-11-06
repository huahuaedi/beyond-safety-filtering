#!/bin/bash
# Usage: bash scripts/hpc/submit_offline_scaled.sh smoke|full|arms [RUN_DIR]
# smoke: manifest build (CPU) -> 2-update GPU smoke on 2 held-out scenes. Review before running full.
# arms : reuse RUN_DIR (needs manifest_v2.json, and a COMPLETE smoke run you checked) -> 7 arms -> summary.
# full : manifest build -> smoke -> 7 arms (baseline, 3 trained modes x 2 seeds) -> paired summary.
set -euo pipefail
STAGE="${1:?smoke|full}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
cd "$ROOT"
summary_job() {
  sbatch --parsable --dependency=afterany:$1 --job-name=scaled_summary --account=xinxia0 --partition=standard --time=00:20:00 \
    --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
    --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $ROOT/tools/rl_prep && python summarize_offline_scaled.py $RUN_DIR"
}
if [ "$STAGE" = "arms" ]; then
  RUN_DIR="${2:?arms needs an existing RUN_DIR}"
  [ -f "$RUN_DIR/manifest_v2.json" ] || { echo "no manifest_v2.json in $RUN_DIR"; exit 1; }
  [ -d "$RUN_DIR/smoke_conditioned/validation" ] || { echo "smoke has not completed in $RUN_DIR"; exit 1; }
  export SCALED_RUN_DIR="$RUN_DIR"
  J3=$(sbatch --parsable --array=0-6%4 scripts/hpc/submit_offline_scaled_train.slurm)
  echo "arms array $J3"; echo "summary job $(summary_job $J3)"; echo "run dir: $RUN_DIR"
  exit 0
fi
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$ROOT/output/offline_scaled_$STAMP"
[ -e "$RUN_DIR" ] && { echo "refusing existing $RUN_DIR"; exit 1; }
mkdir -p "$RUN_DIR"
E="$ROOT/navsim_data/exp"
A="$ROOT/data/navsim/processed_data/anchor"
MANIFEST_CMD="module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; export PATH=\$CONDA_PREFIX/bin:\$PATH; export PYTHONPATH=$ROOT/tools/rl_prep; cd $ROOT && python -m unittest discover -s tools/rl_prep -p test_log_disjoint_offline.py && python tools/rl_prep/build_log_disjoint_manifest.py --root $E/world_action_full_v1 --metric-cache-dirs $E/metric_cache_trainval_rl $E/metric_cache_trainval_rl_eval $E/metric_cache_checkpoint_selection_val --final-eval-tokens $A/trainval_rl_eval_tokens_cached.json --val-tokens $A/checkpoint_selection_val_tokens_cached.json --output $RUN_DIR/manifest_v2.json --label-mode topk_safe --topk 2"
J1=$(sbatch --parsable --job-name=scaled_manifest --account=xinxia0 --partition=standard --cpus-per-task=2 --mem=16g --time=01:00:00 \
  --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err --wrap "$MANIFEST_CMD")
echo "manifest job $J1"
export SCALED_RUN_DIR="$RUN_DIR"
J2=$(sbatch --parsable --dependency=afterok:$J1 --time=00:40:00 \
  --export=ALL,SCALED_MODE=conditioned,SCALED_SEED=20,SCALED_NAME=smoke_conditioned,SCALED_UPDATES=2,SCALED_EVAL_SCENES=2,SCALED_CANDIDATES=4 \
  scripts/hpc/submit_offline_scaled_train.slurm)
echo "smoke job $J2   run dir $RUN_DIR"
if [ "$STAGE" = "full" ]; then
  J3=$(sbatch --parsable --dependency=afterok:$J2 --array=0-6%4 scripts/hpc/submit_offline_scaled_train.slurm)
  echo "arms array $J3"
  J4=$(summary_job $J3)
  echo "summary job $J4"
fi
echo "run dir: $RUN_DIR"
