#!/bin/bash
# Build random_topk / action_topk / world_model_routed_topk from nested V2 OOF.
# Offline verification uses collected NAVSIM labels; no GPU or simulator calls.

#SBATCH --job-name=build_wm_manifests
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"

OOF_ROOT="${WM_RISK_RANKER_OUTPUT:-$REPO_ROOT/output/world_action_risk_ranker_cv_nested_v2}"
DATA_ROOT="${WM_ACTION_DATA_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_v1}"
OUTPUT_DIR="${WM_MANIFEST_OUTPUT:-$REPO_ROOT/results_summary/wm_manifests_nested_v2}"
python tools/rl_prep/evaluate_world_action_mining.py "$OOF_ROOT" \
    --folds "${WM_RISK_CV_FOLDS:-5}" --seed "${WM_RISK_SEED:-0}"
python tools/rl_prep/build_verified_risk_manifests.py \
    "$OOF_ROOT" "$DATA_ROOT" "$OUTPUT_DIR" \
    --folds "${WM_RISK_CV_FOLDS:-5}" \
    --ranker-seed "${WM_RISK_SEED:-0}" \
    --budget "${WM_MANIFEST_BUDGET:-8}" \
    --seed "${WM_MANIFEST_SEED:-0}"
