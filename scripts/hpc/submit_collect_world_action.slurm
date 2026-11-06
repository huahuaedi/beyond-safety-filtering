#!/bin/bash
# Collect policy candidates with verified NAVSIM vector labels for training
# and validating action-conditioned world-model latent/dynamics heads.
# Four array shards write disjoint scene-token files and are safely resumable.

#SBATCH --job-name=collect_world_action
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="$HOME/DriveVLA-W0"
CONDA_ENV="drivevla"

module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"

export VLA_ACTION_TOKENIZER="$REPO_ROOT/pretrained_models/fast"
export VLA_VLM_MODEL="$REPO_ROOT/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json"
export VLA_PATH_REPLACEMENTS="/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RL_REWARD_SOURCE=navsim
export NAVSIM_ENV_PYTHON="$HOME/.conda/envs/navsim/bin/python"
export NAVSIM_METRIC_CACHE_PATH="$REPO_ROOT/navsim_data/exp/metric_cache_trainval_rl"

export RL_COLLECT_ONLY=1
export RL_SEED="${RL_SEED:-0}"
export RL_COLLECTION_DIR="${RL_COLLECTION_DIR:-$REPO_ROOT/navsim_data/exp/world_action_v1}"
export RL_COLLECTION_SHARD_ID="$SLURM_ARRAY_TASK_ID"
export RL_COLLECTION_NUM_SHARDS="$SLURM_ARRAY_TASK_COUNT"
export RL_NUM_STEPS="${RL_NUM_STEPS:-100000}"
export RL_GROUP_SIZE="${RL_GROUP_SIZE:-64}"
export RL_DENOISE_STEPS=10
export RL_NOISE_STD=0.1
export RL_KL_COEF=1.0
export RL_ADVANTAGE_MODE=reference_anchored
export RL_DEPLOYMENT_PROBES=2
export RL_CHECKPOINT_DIR=""

echo "collection shard=$RL_COLLECTION_SHARD_ID/$RL_COLLECTION_NUM_SHARDS"
echo "output=$RL_COLLECTION_DIR candidates_per_scene=$RL_GROUP_SIZE"
srun python tools/rl_prep/rl_stage1_prototype.py
