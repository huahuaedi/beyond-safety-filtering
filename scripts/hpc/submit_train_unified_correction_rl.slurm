#!/bin/bash
# Full-data unified WM outcome + constrained correction-RL training.

#SBATCH --job-name=unified_corr_rl
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/tools/rl_prep:$REPO_ROOT:${PYTHONPATH:-}"
export UNIFIED_SEED="${UNIFIED_SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
export UNIFIED_DATA_ROOT="${UNIFIED_DATA_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_full_v1}"
export UNIFIED_LATENT_ROOT="${UNIFIED_LATENT_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_latents_full_v1}"
export UNIFIED_CONTROL_ROOT="${UNIFIED_CONTROL_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_controls_full_v1}"
export UNIFIED_OUTPUT="${UNIFIED_OUTPUT:-$REPO_ROOT/output/unified_correction_rl_v1/seed_${UNIFIED_SEED}}"
export UNIFIED_OUTCOME_EPOCHS="${UNIFIED_OUTCOME_EPOCHS:-12}"
export UNIFIED_WARMUP_EPOCHS="${UNIFIED_WARMUP_EPOCHS:-4}"
export UNIFIED_RL_EPOCHS="${UNIFIED_RL_EPOCHS:-8}"
export UNIFIED_BATCH_SIZE="${UNIFIED_BATCH_SIZE:-2}"
export UNIFIED_WORKERS="${UNIFIED_WORKERS:-8}"
export UNIFIED_CANDIDATES_PER_SCENE="${UNIFIED_CANDIDATES_PER_SCENE:-8}"
export UNIFIED_MAX_SCENES="${UNIFIED_MAX_SCENES:-0}"
export UNIFIED_OUTCOME_ONLY="${UNIFIED_OUTCOME_ONLY:-0}"

echo "UNIFIED seed=$UNIFIED_SEED outcome_only=$UNIFIED_OUTCOME_ONLY outcome_epochs=$UNIFIED_OUTCOME_EPOCHS warmup_epochs=$UNIFIED_WARMUP_EPOCHS rl_epochs=$UNIFIED_RL_EPOCHS max_scenes=$UNIFIED_MAX_SCENES output=$UNIFIED_OUTPUT"
"$HOME/.conda/envs/drivevla/bin/python" -m unittest \
  tools.rl_prep.test_differentiable_vehicle_model \
  tools.rl_prep.test_continuous_scene_outcome_critic \
  tools.rl_prep.test_unified_correction_policy \
  tools.rl_prep.test_unified_correction_rl
srun "$HOME/.conda/envs/drivevla/bin/python" \
  tools/rl_prep/train_unified_correction_rl.py
