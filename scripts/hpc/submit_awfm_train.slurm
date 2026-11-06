#!/bin/bash
#SBATCH --job-name=awfm
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
# Per-scene KL-budgeted, gate-aware advantage-weighted flow matching (tools/rl_prep/budgeted_awfm.py), same data,
# optimiser, expert anchor and held-out evaluation as the pair_pref / filtered arms.
# Arms (array index), seeds 20,21 each: 0,1 eps 0.1 ; 2,3 eps 0.3 ; 4,5 eps 1.0 (dose curve);
#   6,7 eps 0 (self-distillation control) ; 8,9 eps 0.3 + global velocity preservation ; 10,11 eps 0.3 + targeted preservation (gate passers).
# Required env: SCALED_RUN_DIR (contains manifest_v2.json).
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
export NAVSIM_METRIC_CACHE_PATH="$ROOT/navsim_data/exp/metric_cache_trainval_rl"
RUN_DIR="${SCALED_RUN_DIR:?set SCALED_RUN_DIR}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
EPS=(0.1 0.1 0.3 0.3 1.0 1.0 0 0 0.3 0.3 0.3 0.3)
TAGS=(e0p1 e0p1 e0p3 e0p3 e1p0 e1p0 e0p0 e0p0 gloe0p3 gloe0p3 tgte0p3 tgte0p3)
PRES=(none none none none none none none none global global targeted targeted)
SEEDS=(20 21 20 21 20 21 20 21 20 21 20 21)
EPS_="${AWFM_EPS:-${EPS[$TASK]}}"
TAG="${AWFM_TAG:-${TAGS[$TASK]}}"
SEED="${SCALED_SEED:-${SEEDS[$TASK]}}"
PRESERVE="${AWFM_PRESERVE:-${PRES[$TASK]}}"
OUTNAME="${SCALED_NAME:-${AWFM_OUTPUT_PREFIX:-scaled_awfm_v2_}awfm${TAG}_seed${SEED}}"
ARGS=(--base "${AWFM_BASE:?submit through submit_awfm.sh}" --mode awfm --seed "$SEED" --eps-unsafe "$EPS_" --eps-safe "${AWFM_EPS_SAFE:-0}" --utility-weight "${AWFM_UTILITY:-0}"
 --preserve "$PRESERVE" --preserve-weight "${AWFM_PRESERVE_WEIGHT:-20}"
 --manifest "$RUN_DIR/manifest_v2.json" --output "$RUN_DIR/$OUTNAME"
 --candidates-per-scene "${AWFM_K:-8}"
 --eval-set heldout --eval-scenes "${SCALED_EVAL_SCENES:-600}"
 --max-scenes 0 --epochs "${AWFM_EPOCHS:-1}" --max-updates "${SCALED_UPDATES:-800}" --lr "${SCALED_LR:-1e-5}")
PY="$HOME/.conda/envs/drivevla/bin/python"
# These tests must execute (not skip) in the actual torch training environment.
"$PY" -c 'import torch'
(cd tools/rl_prep && "$PY" -m unittest test_budgeted_awfm test_awfm_preservation test_offline_verified_training test_analyze_repair_transfer test_summarize_offline_scaled test_awfm_validation)
srun "$PY" -u tools/rl_prep/train_offline_verified_policy.py "${ARGS[@]}"

CHECK_ARGS=("$RUN_DIR/$OUTNAME" --preserve "$PRESERVE")
if [ "${AWFM_SMOKE:-0}" = 1 ]; then CHECK_ARGS+=(--expected-scenes 2 --expected-updates 2); fi
"$PY" tools/rl_prep/check_awfm_run.py "${CHECK_ARGS[@]}"
