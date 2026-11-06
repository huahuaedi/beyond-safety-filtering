#!/bin/bash
#SBATCH --job-name=nt_policy_eval
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=05:00:00
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err
# Score saved action-expert checkpoints (and the unmodified base policy) on the frozen navtest cohort with the
# evaluator of the offline training script, so every arm and the baseline use the identical procedure.
# Required env: EVAL_LIST (file with lines "label|path/to/action_policy.pt" or "label|-"), EVAL_OUT (output root).
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
DATA_REPO="${EVAL_DATA_REPO:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
COHORT="${NT_COHORT:-$ASSET/pretrained_models/DriveVLA-W0/posttrain_runs/rl_pareto_navtest_v2_20261005_133703_37354}"
LIST="${EVAL_LIST:?set EVAL_LIST}"
OUTROOT="${EVAL_OUT:?set EVAL_OUT}"
# EVAL_SHARDS=N: every list line is evaluated in N shards; array task t -> line t/N, shard t%N (merge with merge_policy_eval_shards.py)
SHARDS="${EVAL_SHARDS:-1}"
LINE_NO=$((SLURM_ARRAY_TASK_ID / SHARDS + 1)); SHARD=$((SLURM_ARRAY_TASK_ID % SHARDS))
LINE=$(sed -n "${LINE_NO}p" "$LIST")
[ -n "$LINE" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $LIST" >&2; exit 2; }
LABEL="${LINE%%|*}"; CKPT="${LINE#*|}"
test -f "$COHORT/test.pkl" && test -f "$COHORT/test_tokens.json" && test -d "$COHORT/cache"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$ROOT"
export PYTHONPATH="$ROOT/tools/rl_prep:$ROOT/inference/navsim/navsim:$ROOT/utils:$ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT="$DATA_REPO/navsim_data/maps"
export NAVSIM_ENV_PYTHON="$HOME/.conda/envs/navsim/bin/python"
export NAVSIM_METRIC_CACHE_PATH="$COHORT/cache"
export VLA_ACTION_TOKENIZER="$DATA_REPO/pretrained_models/fast"
export VLA_VLM_MODEL="$DATA_REPO/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$DATA_REPO/configs/normalizer_navsim_trainval/norm_stats.json"
export VLA_PATH_REPLACEMENTS="/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:$DATA_REPO/data/navsim/processed_data/data/navsim/processed_data/;/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:$DATA_REPO/data/navsim/processed_data/data/navsim/processed_data/;/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:$DATA_REPO/data/navsim/processed_data/data/navsim/processed_data/"
PY="$HOME/.conda/envs/drivevla/bin/python"
OUTNAME="$LABEL"; [ "$SHARDS" -eq 1 ] || OUTNAME="${LABEL}__shard${SHARD}"
ARGS=(--scene-pkl "$COHORT/test.pkl" --tokens-json "$COHORT/test_tokens.json" --output "$OUTROOT/$OUTNAME" --label "$LABEL" --shard-id "$SHARD" --num-shards "$SHARDS")
[ "$CKPT" = "-" ] || ARGS+=(--checkpoint "$CKPT")
[ -z "${EVAL_LIMIT:-}" ] || ARGS+=(--limit "$EVAL_LIMIT")
[ -z "${EVAL_LATENT_TAG:-}" ] || ARGS+=(--latent-tag "$EVAL_LATENT_TAG")   # independent initial-noise draw; labels must then differ from the default-tag runs
echo "label=$LABEL checkpoint=$CKPT shard=$SHARD/$SHARDS out=$OUTROOT/$OUTNAME"
srun "$PY" -u tools/rl_prep/eval_action_policy.py "${ARGS[@]}"
