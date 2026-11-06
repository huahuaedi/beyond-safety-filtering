#!/bin/bash
# Evaluate all held-out candidates, including those excluded from supervision.

#SBATCH --job-name=control_repair
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"

export CONTROL_REPAIR_SEED="${CONTROL_REPAIR_SEED:-0}"
export CONTROL_DATA_ROOT="${CONTROL_DATA_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_full_v1}"
export CONTROL_SIDECAR_ROOT="${CONTROL_SIDECAR_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_controls_full_v1}"
export CONTROL_ADAPTER_ROOT="${CONTROL_ADAPTER_ROOT:-$REPO_ROOT/output/waypoint_control_adapter_v1}"
export CONTROL_REPAIR_OUTPUT="${CONTROL_REPAIR_OUTPUT:-$REPO_ROOT/output/waypoint_control_adapter_v1/seed_${CONTROL_REPAIR_SEED}/repair_metrics.json}"

srun python tools/rl_prep/evaluate_waypoint_control_repair.py \
  --data-root "$CONTROL_DATA_ROOT" \
  --control-root "$CONTROL_SIDECAR_ROOT" \
  --checkpoint "$CONTROL_ADAPTER_ROOT/seed_${CONTROL_REPAIR_SEED}/model.pt" \
  --output "$CONTROL_REPAIR_OUTPUT" \
  --split test
