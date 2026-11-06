#!/bin/bash
# Collect SFT candidates (16 per scene, noise 0.1) with real NAVSIM scores for the FROZEN navtest cohort,
# in the same npz format as world_action_full_v1. Same collection mode and settings as
# submit_collect_world_action.slurm; only the scene source changes.
#
# The training script reads its scene records from <ASSET_ROOT>/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl.
# Instead of editing it, we point DRIVEVLA_ASSET_ROOT to a small directory in which that file name is a symlink to the
# cohort's test.pkl. The policy checkpoint is passed explicitly, so nothing else depends on that directory.
#
# Required env: NT_OUT (output root), NT_COHORT (frozen cohort run dir with test.pkl, test_tokens.json, cache/)
# Optional: NT_STEPS (scenes per shard, default all), NT_GROUP (candidates per scene, default 16)

#SBATCH --job-name=nt_candidates
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
OUT="${NT_OUT:?set NT_OUT}"
COHORT="${NT_COHORT:?set NT_COHORT}"
test -f "$COHORT/test.pkl" && test -f "$COHORT/test_tokens.json" && test -d "$COHORT/cache"
# The collection code must honour RL_TRAIN_TOKENS and DRIVEVLA_ASSET_ROOT. Older copies (e.g. ~/DriveVLA-W0 from
# August) silently ignore both and collect the whole training pool, so refuse them.
for key in RL_TRAIN_TOKENS DRIVEVLA_ASSET_ROOT RL_COLLECTION_SHARD_ID; do
    grep -q "$key" "$REPO_ROOT/tools/rl_prep/rl_stage1_prototype.py" \
        || { echo "REPO_ROOT=$REPO_ROOT: rl_stage1_prototype.py does not support $key; use a recent repo copy" >&2; exit 3; }
done

module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"
# Same NAVSIM scoring environment as the post-training jobs (job.sh of rl_safety_order_v1_*). Without the maps
# root the scoring subprocess cannot build the PDM scorer.
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT="$REPO_ROOT/navsim_data/maps"
export PYTHONPATH="$REPO_ROOT/inference/navsim/navsim:$REPO_ROOT/utils:$REPO_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

FAKE_ASSETS="$OUT/fake_asset_root"
mkdir -p "$FAKE_ASSETS/pretrained_models/DriveVLA-W0"
ln -sfn "$COHORT/test.pkl" "$FAKE_ASSETS/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
# rl_stage1_prototype.py also reads <ASSET_ROOT>/data/navsim/processed_data/anchor/cluster_centers_8192.npy at start-up.
ln -sfn "$REPO_ROOT/data" "$FAKE_ASSETS/data"
test -f "$FAKE_ASSETS/data/navsim/processed_data/anchor/cluster_centers_8192.npy" \
    || { echo "missing $REPO_ROOT/data/navsim/processed_data/anchor/cluster_centers_8192.npy" >&2; exit 4; }

export DRIVEVLA_ASSET_ROOT="$FAKE_ASSETS"
# NT_POLICY: another action-expert checkpoint to sample candidates from (default: the SFT policy)
export RL_POLICY_CHECKPOINT="${NT_POLICY:-$REPO_ROOT/pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2}"
test -f "$RL_POLICY_CHECKPOINT/config.json" || { echo "no config.json in $RL_POLICY_CHECKPOINT" >&2; exit 5; }
export VLA_ACTION_TOKENIZER="$REPO_ROOT/pretrained_models/fast"
export VLA_VLM_MODEL="$REPO_ROOT/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json"
export VLA_PATH_REPLACEMENTS="/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export RL_REWARD_SOURCE=navsim
export NAVSIM_ENV_PYTHON="$HOME/.conda/envs/navsim/bin/python"
export NAVSIM_METRIC_CACHE_PATH="$COHORT/cache"
export RL_TRAIN_TOKENS="$COHORT/test_tokens.json"

export RL_COLLECT_ONLY=1
export RL_SEED="${RL_SEED:-0}"
export RL_COLLECTION_DIR="$OUT/collection"
export RL_COLLECTION_SHARD_ID="$SLURM_ARRAY_TASK_ID"
export RL_COLLECTION_NUM_SHARDS="$SLURM_ARRAY_TASK_COUNT"
export RL_NUM_STEPS="${NT_STEPS:-100000}"
export RL_GROUP_SIZE="${NT_GROUP:-16}"
export RL_DENOISE_STEPS=10
export RL_NOISE_STD=0.1
export RL_KL_COEF=1.0
export RL_ADVANTAGE_MODE=reference_anchored
export RL_DEPLOYMENT_PROBES=2
export RL_CHECKPOINT_DIR=""

echo "shard=$RL_COLLECTION_SHARD_ID/$RL_COLLECTION_NUM_SHARDS cohort=$COHORT out=$RL_COLLECTION_DIR group=$RL_GROUP_SIZE steps=$RL_NUM_STEPS"
srun python tools/rl_prep/rl_stage1_prototype.py
