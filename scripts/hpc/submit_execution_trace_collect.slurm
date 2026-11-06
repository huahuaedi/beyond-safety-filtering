#!/bin/bash
# Stage 1 of the execution-aware RL plan: sample SFT-policy candidates and
# record, per NAVSIM scoring call, the desired and executed (LQR + bicycle)
# trajectories, tracking error and drivable margins. Scores are unchanged.
# Submitted by scripts/hpc/submit_execution_trace_pipeline.sh (sets
# TRACE_RUN_ROOT). Shards are resumable: finished scenes are skipped.

#SBATCH --job-name=execution_trace
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
: "${TRACE_RUN_ROOT:?set by submit_execution_trace_pipeline.sh}"

REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
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
export RL_COLLECTION_DIR="$TRACE_RUN_ROOT/collection"
export RL_EXECUTION_TRACE_DIR="$TRACE_RUN_ROOT/traces"
export RL_EXECUTION_TRACE_VERIFY="${TRACE_VERIFY:-0}"
export RL_COLLECTION_SHARD_ID="$SLURM_ARRAY_TASK_ID"
export RL_COLLECTION_NUM_SHARDS="$SLURM_ARRAY_TASK_COUNT"
export RL_NUM_STEPS="${TRACE_SCENES_PER_SHARD:-500}"
export RL_GROUP_SIZE="${TRACE_GROUP_SIZE:-16}"
export RL_DENOISE_STEPS=10
export RL_NOISE_STD=0.1
export RL_KL_COEF=1.0
export RL_ADVANTAGE_MODE=reference_anchored
export RL_DEPLOYMENT_PROBES=2
export RL_CHECKPOINT_DIR=""

"$NAVSIM_ENV_PYTHON" -c 'import shapely; assert int(shapely.__version__.split(".")[0]) >= 2, shapely.__version__; print("navsim shapely", shapely.__version__)'
echo "shard=$RL_COLLECTION_SHARD_ID/$RL_COLLECTION_NUM_SHARDS scenes<=$RL_NUM_STEPS G=$RL_GROUP_SIZE verify=$RL_EXECUTION_TRACE_VERIFY"
echo "policy=${RL_POLICY_CHECKPOINT:-default SFT} traces=$RL_EXECUTION_TRACE_DIR"
srun python tools/rl_prep/rl_stage1_prototype.py
