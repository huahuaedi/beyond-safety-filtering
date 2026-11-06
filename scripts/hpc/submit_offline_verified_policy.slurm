#!/bin/bash
#SBATCH --job-name=offline_verified_policy
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err
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
export NAVSIM_METRIC_CACHE_PATH="$ROOT/navsim_data/exp/metric_cache_checkpoint_selection_val"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
MODES=(unconditional conditioned)
MODE="${OFFLINE_MODE:-${MODES[$((TASK % 2))]}}"
SEED="${OFFLINE_SEED:-$((20 + TASK / 2))}"
RUN="${OFFLINE_RUN:-pilot_v1}"
ARGS=(--mode "$MODE" --seed "$SEED"
 --manifest "$ROOT/output/offline_verified_20260907/audit.json"
 --output "$ROOT/output/offline_verified_20260907/${RUN}_${MODE}_seed${SEED}"
 --max-scenes "${OFFLINE_MAX_SCENES:-64}"
 --max-updates "${OFFLINE_MAX_UPDATES:-128}"
 --eval-scenes "${OFFLINE_EVAL_SCENES:-150}"
 --candidates-per-scene "${OFFLINE_CANDIDATES:-8}")
"$HOME/.conda/envs/drivevla/bin/python" -m unittest discover -s tools/rl_prep -p test_offline_verified_data.py
srun "$HOME/.conda/envs/drivevla/bin/python" -u tools/rl_prep/train_offline_verified_policy.py "${ARGS[@]}"
