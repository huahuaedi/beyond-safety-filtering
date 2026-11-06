#!/bin/bash
# Feature-block risk-ranker ablations: 3 seeds x 5 outer folds x 7 modes.
# Each task also scores its model on the disjoint fresh scenes.

#SBATCH --job-name=wm_baselines_cv
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --array=0-104%8
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"

MODES=(action geom action_geom action_scene1 resid1 action_geom_wm1 action_geom_resid1)
FOLDS=5
SEEDS=3
NM=${#MODES[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge $((SEEDS * FOLDS * NM)) ]; then
    echo "array index $SLURM_ARRAY_TASK_ID beyond $((SEEDS * FOLDS * NM)) tasks" >&2; exit 2
fi
export WM_BASELINE_MODE="${MODES[$((SLURM_ARRAY_TASK_ID % NM))]}"
export WM_RISK_CV_FOLD="$(( (SLURM_ARRAY_TASK_ID / NM) % FOLDS ))"
export WM_RISK_SEED="$(( SLURM_ARRAY_TASK_ID / (NM * FOLDS) ))"
export WM_RISK_CV_FOLDS="$FOLDS"

EXP="$REPO_ROOT/navsim_data/exp"
export WM_ACTION_DATA_ROOT="${WM_ACTION_DATA_ROOT:-$EXP/world_action_v1}"
export WM_LATENT_ROOT="${WM_LATENT_ROOT:-$EXP/world_action_latents_v1}"
export WM_GEOM_ROOT="${WM_GEOM_ROOT:-$EXP/world_action_geom_v1}"
export WM_FRESH_DATA_ROOT="${WM_FRESH_DATA_ROOT:-$EXP/world_action_full_v1}"
export WM_FRESH_LATENT_ROOT="${WM_FRESH_LATENT_ROOT:-$EXP/world_action_latents_full_v1}"
export WM_FRESH_GEOM_ROOT="${WM_FRESH_GEOM_ROOT:-$EXP/world_action_geom_full_v1}"
: "${WM_RISK_RANKER_OUTPUT:?set WM_RISK_RANKER_OUTPUT to a fresh output directory}"
export WM_RISK_EPOCHS="${WM_RISK_EPOCHS:-40}"
export WM_RISK_SCENES_PER_BATCH="${WM_RISK_SCENES_PER_BATCH:-4}"

echo "mode=$WM_BASELINE_MODE fold=$WM_RISK_CV_FOLD seed=$WM_RISK_SEED"
srun python tools/rl_prep/train_risk_baselines.py
