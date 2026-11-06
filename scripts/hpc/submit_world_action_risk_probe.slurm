#!/bin/bash
# Compare action-only and multi-depth world-model latent risk probes.

#SBATCH --job-name=wm_risk_probe
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --array=0-5
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

MODES=(action layer0 layer1 layer2 layer3 latent_all)
export WM_RISK_FEATURE_MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
export WM_ACTION_DATA_ROOT="$REPO_ROOT/navsim_data/exp/world_action_v1"
export WM_LATENT_ROOT="$REPO_ROOT/navsim_data/exp/world_action_latents_v1"
export WM_RISK_PROBE_OUTPUT="$REPO_ROOT/output/world_action_risk_probe_v1"
export WM_RISK_SEED="${WM_RISK_SEED:-0}"
export WM_RISK_EPOCHS="${WM_RISK_EPOCHS:-30}"
export WM_RISK_BATCH_SIZE="${WM_RISK_BATCH_SIZE:-256}"

echo "mode=$WM_RISK_FEATURE_MODE seed=$WM_RISK_SEED"
srun python tools/rl_prep/train_world_action_risk_probe.py

