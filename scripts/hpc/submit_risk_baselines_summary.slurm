#!/bin/bash
# Paired-bootstrap comparison of all modes + geometric-feature audit.

#SBATCH --job-name=wm_baselines_sum
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"

EXP="$REPO_ROOT/navsim_data/exp"
: "${WM_RISK_RANKER_OUTPUT:?}"
: "${WM_SUMMARY_OUTPUT:?set WM_SUMMARY_OUTPUT to a new summary json path}"
python tools/rl_prep/summarize_risk_baselines.py \
    --oof-root "$WM_RISK_RANKER_OUTPUT" \
    --modes action geom action_geom action_scene1 resid1 action_geom_wm1 action_geom_resid1 \
    --reference action_geom --seeds 0 1 2 --folds 5 \
    --data-root "${WM_FRESH_DATA_ROOT:-$EXP/world_action_full_v1}" \
    --geom-root "${WM_FRESH_GEOM_ROOT:-$EXP/world_action_geom_full_v1}" \
    --output "$WM_SUMMARY_OUTPUT"
