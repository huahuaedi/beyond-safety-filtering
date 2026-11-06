#!/bin/bash
#SBATCH --job-name=offline_scaled
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=04:30:00
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err
# Arms (array index): 0 baseline, 1-2 unconditional s20/s21, 3-4 conditioned, 5-6 filtered.
# Required env: SCALED_RUN_DIR (contains manifest_v2.json). Optional: SCALED_UPDATES, SCALED_LR, SCALED_EVAL_SCENES.
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
cd "$ROOT"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT/tools/rl_prep:$ROOT/inference/navsim/navsim:$ROOT/utils:$ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NAVSIM_ENV_PYTHON="$HOME/.conda/envs/navsim/bin/python"
# this cache covers every scene of world_action_full_v1 (checked 5654/751/766)
export NAVSIM_METRIC_CACHE_PATH="$ROOT/navsim_data/exp/metric_cache_trainval_rl"
RUN_DIR="${SCALED_RUN_DIR:?set SCALED_RUN_DIR}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
MODES=(baseline unconditional unconditional conditioned conditioned filtered filtered)
SEEDS=(20 20 21 20 21 20 21)
MODE="${SCALED_MODE:-${MODES[$TASK]}}"
SEED="${SCALED_SEED:-${SEEDS[$TASK]}}"
NAME="${SCALED_NAME:-scaled_v1_${MODE}_seed${SEED}}"
ARGS=(--mode "$MODE" --seed "$SEED"
 --manifest "$RUN_DIR/manifest_v2.json"
 --output "$RUN_DIR/$NAME"
 --label-mode topk_safe --topk 2 --positive-fraction 0.5
 --eval-set heldout --eval-scenes "${SCALED_EVAL_SCENES:-600}"
 --max-scenes 0 --candidates-per-scene "${SCALED_CANDIDATES:-2}"
 --max-updates "${SCALED_UPDATES:-800}" --lr "${SCALED_LR:-1e-5}")
PY="$HOME/.conda/envs/drivevla/bin/python"
"$PY" -m unittest discover -s tools/rl_prep -p "test_log_disjoint_offline.py"
srun "$PY" -u tools/rl_prep/train_offline_verified_policy.py "${ARGS[@]}"
