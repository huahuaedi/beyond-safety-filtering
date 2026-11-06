#!/bin/bash
# Top-K anchor candidates of the anchor-scoring planner (Emu3_Multi_Anchor_PDMS_90.3), scored by the simulator.
# Env: AN_TOKENS (json list), AN_SCENE_PKL, AN_CACHE (metric cache), AN_OUT (collection root), optional AN_K (16), AN_LIMIT.
#SBATCH --job-name=anchor_cand
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err
set -euo pipefail
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
OFFLINE="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
HUB="${AN_HUB:-$ASSET/pretrained_models/DriveVLA-W0/Emu3_Multi_Anchor_PDMS_90.3}"
test -f "$HUB/config.json" && ls "$HUB"/model-*.safetensors > /dev/null
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"
export DRIVEVLA_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/inference/navsim/navsim:$REPO_ROOT/utils:$REPO_ROOT:${PYTHONPATH:-}"
export VLA_ACTION_TOKENIZER="$REPO_ROOT/pretrained_models/fast" VLA_VLM_MODEL="$REPO_ROOT/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json"
export VLA_ANCHOR_ENABLED=true VLA_ANCHOR_CLUSTER_PATH="$REPO_ROOT/data/navsim/processed_data/anchor/cluster_centers_8192.npy"
export VLA_ANCHOR_METRIC_SCORE_PATH="$REPO_ROOT/data/navsim/processed_data/anchor/formatted_pdm_score_8192.npy"
export VLA_PATH_REPLACEMENTS="/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT="$REPO_ROOT/navsim_data/maps"
export NAVSIM_ENV_PYTHON="$HOME/.conda/envs/navsim/bin/python" NAVSIM_METRIC_CACHE_PATH="${AN_CACHE:?AN_CACHE}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXTRA=(); [ "${AN_LIMIT:-0}" -gt 0 ] && EXTRA+=(--limit "$AN_LIMIT")
srun python "$OFFLINE/tools/rl_prep/collect_anchor_candidates.py" --emu-hub "$HUB" --scene-pkl "${AN_SCENE_PKL:?AN_SCENE_PKL}" \
    --tokens-json "${AN_TOKENS:?AN_TOKENS}" --output "${AN_OUT:?AN_OUT}" --k "${AN_K:-16}" \
    --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards "$SLURM_ARRAY_TASK_COUNT" ${EXTRA[@]+"${EXTRA[@]}"}
