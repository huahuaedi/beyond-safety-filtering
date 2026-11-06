#!/bin/bash
# Phase-1-only diagnostic run: checks whether gate_prob_needs_correction vs.
# gate_prob_safe show any discrimination at all, without paying for phase 2
# (joint RL, ~16h) or eval. Throwaway seed=99, output goes to its own dir so
# it never collides with a real numbered-seed run.

#SBATCH --job-name=phase1_diag
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
SEED=99
WARM_ROOT="$REPO_ROOT/output/unified_drivevla_warm/phase1_diagnostic_seed${SEED}"

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

echo "=== phase 1 only (diagnostic): fit unified outcome and warm+RL correction policy, seed=$SEED ==="
srun "$HOME/.conda/envs/drivevla/bin/python" \
  tools/rl_prep/train_unified_correction_rl.py
echo "=== phase 1 diagnostic done -- not running phase 2 ==="
