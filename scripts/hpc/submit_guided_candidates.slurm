#!/bin/bash
# Candidates sampled with geometric cost guidance (GD_LAMBDA > 0) or, for lambda = 0, by the unchanged sampler (control).
# Same collect-only loop, policy, noise seed and scoring as submit_navtest_candidates.slurm; only the candidate rollouts are guided.
# Env: GD_TOKENS (json list), GD_SCENE_PKL (records of those scenes), GD_CACHE (metric cache), GD_GUIDE_DIR (extract_guidance_inputs output),
#      GD_OUT (output root), GD_LAMBDA, optional GD_POWER, NT_STEPS, RL_SEED, NT_POLICY.
#SBATCH --job-name=guided_cand
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
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
OFFLINE="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
OUT="${GD_OUT:?GD_OUT}"; TOKENS="${GD_TOKENS:?GD_TOKENS}"; PKL="${GD_SCENE_PKL:?GD_SCENE_PKL}"; CACHE="${GD_CACHE:?GD_CACHE}"
GUIDE_DIR="${GD_GUIDE_DIR:?GD_GUIDE_DIR}"; LAMBDA="${GD_LAMBDA:?GD_LAMBDA}"
for key in RL_TRAIN_TOKENS DRIVEVLA_ASSET_ROOT RL_COLLECTION_SHARD_ID; do
    grep -q "$key" "$REPO_ROOT/tools/rl_prep/rl_stage1_prototype.py" || { echo "old repo copy: no $key" >&2; exit 3; }
done
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT="$REPO_ROOT/navsim_data/maps"
OVERLAY="$OUT/overlay"; mkdir -p "$OVERLAY"
cp "$OFFLINE/tools/rl_prep/collect_guided_candidates.py" "$OFFLINE/tools/rl_prep/guidance_geometry.py" "$OVERLAY/"
export PYTHONPATH="$REPO_ROOT/tools/rl_prep:$REPO_ROOT/inference/navsim/navsim:$REPO_ROOT/utils:$REPO_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
FAKE="$OUT/fake_asset_root"; mkdir -p "$FAKE/pretrained_models/DriveVLA-W0"
ln -sfn "$PKL" "$FAKE/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
ln -sfn "$REPO_ROOT/data" "$FAKE/data"
export DRIVEVLA_ASSET_ROOT="$FAKE"
export RL_POLICY_CHECKPOINT="${NT_POLICY:-$REPO_ROOT/pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2}"
export VLA_ACTION_TOKENIZER="$REPO_ROOT/pretrained_models/fast" VLA_VLM_MODEL="$REPO_ROOT/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json"
export VLA_PATH_REPLACEMENTS="/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RL_REWARD_SOURCE=navsim NAVSIM_ENV_PYTHON="$HOME/.conda/envs/navsim/bin/python" NAVSIM_METRIC_CACHE_PATH="$CACHE" RL_TRAIN_TOKENS="$TOKENS"
export RL_COLLECT_ONLY=1 RL_SEED="${RL_SEED:-0}" RL_COLLECTION_DIR="$OUT/collection"
export RL_COLLECTION_SHARD_ID="$SLURM_ARRAY_TASK_ID" RL_COLLECTION_NUM_SHARDS="$SLURM_ARRAY_TASK_COUNT"
export RL_NUM_STEPS="${NT_STEPS:-100000}" RL_GROUP_SIZE=16 RL_DENOISE_STEPS=10 RL_NOISE_STD=0.1 RL_KL_COEF=1.0
export RL_ADVANTAGE_MODE=reference_anchored RL_DEPLOYMENT_PROBES=2 RL_CHECKPOINT_DIR=""
export GUIDE_DIR="$GUIDE_DIR" GUIDE_LAMBDA="$LAMBDA" GUIDE_POWER="${GD_POWER:-1.0}"
echo "shard=$RL_COLLECTION_SHARD_ID/$RL_COLLECTION_NUM_SHARDS lambda=$LAMBDA out=$RL_COLLECTION_DIR"
srun python "$OVERLAY/collect_guided_candidates.py"
