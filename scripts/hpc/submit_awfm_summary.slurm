#!/bin/bash
#SBATCH --job-name=awfm_summary
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:20:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
set -euo pipefail
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
cd "$ROOT/tools/rl_prep"
ARGS=("${SCALED_RUN_DIR:?}" --prefix "${AWFM_OUTPUT_PREFIX:?}" --require-complete
 --metric-cache-dir "$ROOT/navsim_data/exp/metric_cache_trainval_rl"
 --out-name "awfm_v2_${AWFM_REPORT_STAGE:?}_comparison.json")
for arm in ${AWFM_REQUIRED:?}; do ARGS+=(--require-arm "$arm"); done
python summarize_offline_scaled.py "${ARGS[@]}"
