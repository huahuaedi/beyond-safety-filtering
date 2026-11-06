#!/bin/bash
# Full-data action/WM feasibility-transformer ablations; quality/dynamics are labels.

#SBATCH --job-name=wm_feasibility
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"

MODES=(action wm_mean wm_layer full)
export WM_FEASIBILITY_MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
export WM_ACTION_DATA_ROOT="${WM_ACTION_DATA_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_full_v1}"
export WM_LATENT_ROOT="${WM_LATENT_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_latents_full_v1}"
export WM_QUALITY_ROOT="${WM_QUALITY_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_quality_full_v1}"
export WM_DYNAMICS_ROOT="${WM_DYNAMICS_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_dynamics_full_v2}"
export WM_FEASIBILITY_OUTPUT="${WM_FEASIBILITY_OUTPUT:-$REPO_ROOT/output/world_action_feasibility_transformer_v1}"
export WM_FEASIBILITY_SEED="${WM_FEASIBILITY_SEED:-0}"
export WM_FEASIBILITY_EPOCHS="${WM_FEASIBILITY_EPOCHS:-12}"
export WM_FEASIBILITY_BATCH_SIZE="${WM_FEASIBILITY_BATCH_SIZE:-2}"
export WM_FEASIBILITY_WORKERS="${WM_FEASIBILITY_WORKERS:-4}"
export WM_FEASIBILITY_MODEL_DIM="${WM_FEASIBILITY_MODEL_DIM:-256}"
export WM_FEASIBILITY_MAX_SCENES="${WM_FEASIBILITY_MAX_SCENES:-0}"

echo "mode=$WM_FEASIBILITY_MODE seed=$WM_FEASIBILITY_SEED epochs=$WM_FEASIBILITY_EPOCHS"
srun python tools/rl_prep/train_world_action_feasibility_transformer.py
