#!/bin/bash
# One allocation: WM outcome fitting -> identity warm-up -> model-based
# correction RL -> online Reference-Anchored NAVSIM joint policy optimization.

#SBATCH --job-name=unified_drivevla
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
SEED="${SLURM_ARRAY_TASK_ID:-${UNIFIED_SEED:-0}}"
WARM_ROOT="$REPO_ROOT/output/unified_drivevla_warm/seed_${SEED}"
FINAL_ROOT="$REPO_ROOT/pretrained_models/DriveVLA-W0/unified_drivevla_rl_seed${SEED}"

module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/tools/rl_prep:$REPO_ROOT:${PYTHONPATH:-}"

export UNIFIED_SEED="$SEED"
export UNIFIED_OUTPUT="$WARM_ROOT"
export UNIFIED_OUTCOME_EPOCHS="${UNIFIED_OUTCOME_EPOCHS:-12}"
export UNIFIED_WARMUP_EPOCHS="${UNIFIED_WARMUP_EPOCHS:-4}"
export UNIFIED_RL_EPOCHS="${UNIFIED_RL_EPOCHS:-8}"
export UNIFIED_BATCH_SIZE="${UNIFIED_BATCH_SIZE:-2}"
export UNIFIED_WORKERS="${UNIFIED_WORKERS:-8}"
export UNIFIED_CANDIDATES_PER_SCENE="${UNIFIED_CANDIDATES_PER_SCENE:-8}"
export UNIFIED_MAX_SCENES="${UNIFIED_MAX_SCENES:-0}"

echo "=== phase 1/2: fit unified outcome and warm correction policy, seed=$SEED ==="
"$HOME/.conda/envs/drivevla/bin/python" -m unittest \
  tools.rl_prep.test_differentiable_vehicle_model \
  tools.rl_prep.test_continuous_scene_outcome_critic \
  tools.rl_prep.test_unified_correction_policy \
  tools.rl_prep.test_unified_correction_rl
srun "$HOME/.conda/envs/drivevla/bin/python" \
  tools/rl_prep/train_unified_correction_rl.py

echo "=== phase 2/2: joint Action Expert + correction Reference-Anchored RL ==="
export RL_SEED="$SEED"
export RL_NUM_STEPS="${RL_NUM_STEPS:-50}"
export RL_SAVE_STEPS="${RL_SAVE_STEPS:-10,25,40,50}"
export RL_ROLLOUT_MICRO_BATCH="${RL_ROLLOUT_MICRO_BATCH:-1}"
export RL_CONTROL_ADAPTER_CHECKPOINT=""
export RL_UNIFIED_CORRECTION_CHECKPOINT="$WARM_ROOT/correction_policy.pt"
export RL_UNIFIED_OUTCOME_CHECKPOINT="$WARM_ROOT/outcome_critic.pt"
export RL_UNIFIED_LATENT_ROOT="$REPO_ROOT/navsim_data/exp/world_action_latents_full_v1"
export RL_UNIFIED_CORRECTION_LOGP_COEF="${RL_UNIFIED_CORRECTION_LOGP_COEF:-1.0}"
export RL_CHECKPOINT_DIR="$FINAL_ROOT"

# Source the established environment/reward launcher. Its SBATCH directives
# are comments here; this allocation remains the only job.
source "$REPO_ROOT/scripts/hpc/submit_rl_stage1.slurm"
