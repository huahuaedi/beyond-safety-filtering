#!/bin/bash
# Build the RL metric cache from every unique DriveVLA trainval token that can
# be resolved by NAVSIM, while preserving and excluding the existing held-out
# evaluation candidate pool.  This is a CPU-only, one-time preprocessing job.

#SBATCH --job-name=cache_trainval_all
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail

REPO_ROOT="$HOME/DriveVLA-W0"

module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$REPO_ROOT"

export VLA_ACTION_TOKENIZER="$REPO_ROOT/pretrained_models/fast"
export VLA_VLM_MODEL="$REPO_ROOT/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json"

echo "=== [1/2] Build full train candidate pool, preserving held-out eval pool ==="
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
export RL_USE_ALL_TRAIN_SCENES=1
export RL_FREEZE_EVAL_POOL=1
python tools/rl_prep/sample_trainval_scene_pool.py

echo "=== [2/2] Resolve all valid NAVSIM anchors and build/extend metric cache ==="
conda activate navsim
export PATH="$CONDA_PREFIX/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$REPO_ROOT/navsim_data/maps"
export NAVSIM_EXP_ROOT="$REPO_ROOT/navsim_data/exp"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT/inference/navsim/navsim"
export OPENSCENE_DATA_ROOT="$REPO_ROOT/navsim_data"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

RL_CACHE_TARGET=train python tools/rl_prep/cache_trainval_rl_scenes.py

echo "=== Full trainval cache complete; held-out eval artifacts were not modified ==="
