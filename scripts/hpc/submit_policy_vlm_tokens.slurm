#!/bin/bash
#SBATCH --job-name=pv_tokens
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err
# Token-level VLM-stream hidden states of the visual tokens (Stage A). Env: PV_PKL, PV_TOKENS, PV_OUT; PV_SMOKE=1 and PV_LIMIT=N for the layout and cost smoke test (one task).
# Env: PV_PKL (scene pickle with a token per record), PV_TOKENS (json list), PV_OUT (new directory). 8 shards.
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
DATA_REPO="${EVAL_DATA_REPO:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
PKL="${PV_PKL:?PV_PKL}"; TOKENS="${PV_TOKENS:?PV_TOKENS}"; OUT="${PV_OUT:-}"
[ "${PV_SMOKE:-0}" = 1 ] || [ -n "$OUT" ] || { echo "set PV_OUT (or PV_SMOKE=1)"; exit 2; }
test -f "$PKL" && test -f "$TOKENS"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$ROOT"
export PYTHONPATH="$ROOT/tools/rl_prep:$ROOT/inference/navsim/navsim:$ROOT/utils:$ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLA_ACTION_TOKENIZER="$DATA_REPO/pretrained_models/fast"
export VLA_VLM_MODEL="$DATA_REPO/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$DATA_REPO/configs/normalizer_navsim_trainval/norm_stats.json"
export VLA_PATH_REPLACEMENTS="/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:$DATA_REPO/data/navsim/processed_data/data/navsim/processed_data/;/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:$DATA_REPO/data/navsim/processed_data/data/navsim/processed_data/;/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:$DATA_REPO/data/navsim/processed_data/data/navsim/processed_data/"
PY="$HOME/.conda/envs/drivevla/bin/python"
EXTRA=()
if [ "${PV_SMOKE:-0}" = 1 ]; then EXTRA+=(--smoke --limit "${PV_LIMIT:-20}"); else EXTRA+=(--output "$OUT"); fi
# production runs enforce the layout found by the smoke test (2 frames of 18x32, the extractor defaults); PV_NO_CHECK=1 only for a new smoke test
[ "${PV_NO_CHECK:-0}" = 1 ] && EXTRA+=(--no-check)
srun "$PY" -u tools/rl_prep/extract_policy_vlm_tokens.py --scene-pkl "$PKL" --tokens-json "$TOKENS" \
    --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards "$SLURM_ARRAY_TASK_COUNT" "${EXTRA[@]}"
