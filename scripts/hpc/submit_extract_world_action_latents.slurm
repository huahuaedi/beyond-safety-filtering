#!/bin/bash
# Extract multi-layer action-conditioned NuPlan world-model latents.

#SBATCH --job-name=wm_action_latents
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --array=0-3
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

export WM_ACTION_DATA_ROOT="${WM_ACTION_DATA_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_v1}"
export WM_LATENT_ROOT="${WM_LATENT_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_latents_v1}"
export WM_NUPLAN_CKPT="${WM_NUPLAN_CKPT:-$REPO_ROOT/pretrained_models/DriveVLA-W0/Emu3_NuPlan_Pretrain_Cktps}"
export WM_LATENT_BATCH_SIZE="${WM_LATENT_BATCH_SIZE:-8}"
export WM_LATENT_LAYERS="${WM_LATENT_LAYERS:--1,-9,-17,-25}"
export WM_LATENT_MAX_SCENES="${WM_LATENT_MAX_SCENES:-0}"
export WM_LATENT_SHARD_ID="$SLURM_ARRAY_TASK_ID"
export WM_LATENT_NUM_SHARDS="$SLURM_ARRAY_TASK_COUNT"

echo "latent shard=$WM_LATENT_SHARD_ID/$WM_LATENT_NUM_SHARDS"
echo "input=$WM_ACTION_DATA_ROOT output=$WM_LATENT_ROOT"
srun python tools/rl_prep/extract_world_action_latents.py
