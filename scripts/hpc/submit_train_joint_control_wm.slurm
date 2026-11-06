#!/bin/bash
# One allocation trains the continuous risk student and three joint adapters.

#SBATCH --job-name=joint_ctrl_wm
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/tools/rl_prep:$REPO_ROOT:${PYTHONPATH:-}"

: "${JOINT_WM_TEACHER_CHECKPOINT:=$REPO_ROOT/output/world_action_feasibility_transformer_v3/wm_mean/seed_0/model.pt}"
export JOINT_WM_TEACHER_CHECKPOINT
export JOINT_DATA_ROOT="${JOINT_DATA_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_full_v1}"
export JOINT_LATENT_ROOT="${JOINT_LATENT_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_latents_full_v1}"
export JOINT_CONTROL_ROOT="${JOINT_CONTROL_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_controls_full_v1}"
export JOINT_ADAPTER_CHECKPOINT="${JOINT_ADAPTER_CHECKPOINT:-$REPO_ROOT/output/waypoint_control_adapter_v1/seed_0/model.pt}"
export JOINT_OUTPUT="${JOINT_OUTPUT:-$REPO_ROOT/output/joint_control_wm_v1}"
export JOINT_STUDENT_CHECKPOINT="${JOINT_STUDENT_CHECKPOINT:-}"
export JOINT_STUDENT_EPOCHS="${JOINT_STUDENT_EPOCHS:-12}"
export JOINT_ADAPTER_EPOCHS="${JOINT_ADAPTER_EPOCHS:-8}"
export JOINT_BATCH_SIZE="${JOINT_BATCH_SIZE:-2}"
export JOINT_WORKERS="${JOINT_WORKERS:-8}"
export JOINT_RISK_LAMBDAS="${JOINT_RISK_LAMBDAS:-0,0.01,0.05}"
export JOINT_MAX_SCENES="${JOINT_MAX_SCENES:-0}"

echo "student_epochs=$JOINT_STUDENT_EPOCHS adapter_epochs=$JOINT_ADAPTER_EPOCHS lambdas=$JOINT_RISK_LAMBDAS max_scenes=$JOINT_MAX_SCENES"
"$HOME/.conda/envs/drivevla/bin/python" -m unittest \
  tools.rl_prep.test_continuous_scene_risk_critic \
  tools.rl_prep.test_waypoint_control_adapter \
  tools.rl_prep.test_differentiable_vehicle_model
srun "$HOME/.conda/envs/drivevla/bin/python" tools/rl_prep/train_joint_control_wm.py
