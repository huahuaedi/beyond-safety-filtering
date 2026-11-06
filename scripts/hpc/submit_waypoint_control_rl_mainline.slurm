#!/bin/bash
# Final mainline: DriveVLA waypoint policy + selective differentiable control
# adapter + reference-anchored NAVSIM RL. Submit with --array=0-2%3.

#SBATCH --job-name=wp_ctrl_rl
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="$HOME/DriveVLA-W0"
TASK_SEED="${SLURM_ARRAY_TASK_ID:-0}"

# One unambiguous, reproducible configuration. Explicit sbatch exports may
# still override the training length/adapter, but never the array seed or the
# seed-specific output directory.
export RL_SEED="$TASK_SEED"
export RL_NUM_STEPS="${RL_NUM_STEPS:-50}"
export RL_SAVE_STEPS="${RL_SAVE_STEPS:-10,25,40,50}"
export RL_ROLLOUT_MICRO_BATCH="${RL_ROLLOUT_MICRO_BATCH:-1}"
export RL_CONTROL_ADAPTER_CHECKPOINT="${RL_CONTROL_ADAPTER_CHECKPOINT:-$REPO_ROOT/output/waypoint_control_adapter_v1/seed_0/model.pt}"
export RL_CONTROL_ADAPTER_MODE="infeasible_only"
export RL_CONTROL_ADAPTER_MAX_POSITION_RMSE="${RL_CONTROL_ADAPTER_MAX_POSITION_RMSE:-inf}"
export RL_CHECKPOINT_DIR="$REPO_ROOT/pretrained_models/DriveVLA-W0/waypoint_control_ref_anchor_seed${TASK_SEED}"

echo "MAINLINE seed=$RL_SEED steps=$RL_NUM_STEPS adapter=$RL_CONTROL_ADAPTER_CHECKPOINT mode=$RL_CONTROL_ADAPTER_MODE checkpoint_dir=$RL_CHECKPOINT_DIR"

# Source the known-good environment and training launcher. Its #SBATCH lines
# are comments at execution time; resource allocation comes from this wrapper.
source "$REPO_ROOT/scripts/hpc/submit_rl_stage1.slurm"
