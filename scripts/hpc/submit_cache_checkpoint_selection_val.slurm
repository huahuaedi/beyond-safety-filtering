#!/bin/bash
# One-time: builds the NAVSIM metric_cache for the 150-scene
# checkpoint-selection validation subset (navsim env). Prerequisite for
# scoring any checkpoint against that subset via submit_eval_subset.slurm's
# EVAL_METRIC_CACHE_PATH/EVAL_TOKENS_JSON override.

#SBATCH --job-name=cache_ckpt_sel_val
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$REPO_ROOT"

conda activate navsim
export PATH="$CONDA_PREFIX/bin:$PATH"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$REPO_ROOT/navsim_data/maps"
export NAVSIM_EXP_ROOT="$REPO_ROOT/navsim_data/exp"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT/inference/navsim/navsim"
export OPENSCENE_DATA_ROOT="$REPO_ROOT/navsim_data"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

RL_CACHE_TARGET=checkpoint_selection_val python3 tools/rl_prep/cache_trainval_rl_scenes.py
