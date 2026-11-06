#!/bin/bash
# Train lightweight waypoint-to-control adapters; one seed per array task.

#SBATCH --job-name=control_adapter
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx60
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"

export CONTROL_ADAPTER_SEED="${CONTROL_ADAPTER_SEED:-$SLURM_ARRAY_TASK_ID}"
export CONTROL_DATA_ROOT="${CONTROL_DATA_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_full_v1}"
export CONTROL_SIDECAR_ROOT="${CONTROL_SIDECAR_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_controls_full_v1}"
export CONTROL_ADAPTER_OUTPUT="${CONTROL_ADAPTER_OUTPUT:-$REPO_ROOT/output/waypoint_control_adapter_v1}"
export CONTROL_ADAPTER_EPOCHS="${CONTROL_ADAPTER_EPOCHS:-20}"
export CONTROL_ADAPTER_BATCH_SIZE="${CONTROL_ADAPTER_BATCH_SIZE:-4}"
export CONTROL_ADAPTER_WORKERS="${CONTROL_ADAPTER_WORKERS:-8}"
export CONTROL_ADAPTER_MAX_SCENES="${CONTROL_ADAPTER_MAX_SCENES:-0}"

echo "seed=$CONTROL_ADAPTER_SEED epochs=$CONTROL_ADAPTER_EPOCHS max_scenes=$CONTROL_ADAPTER_MAX_SCENES"
srun python tools/rl_prep/train_waypoint_control_adapter.py
