#!/bin/bash
#SBATCH --job-name=pair_pref
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
# Pairwise preference distillation on a frozen-VLM action expert, log-isolated, same evaluation as the scaled arms.
# Arms (array index): 0,1 marginpref s20,s21; 2,3 pdmspref s20,s21; 4,5 nearestpref s20,s21.
# The three pair files share the loser and differ only in the winner (largest margin / highest PDMS / closest to the reference output).
# Required env: SCALED_RUN_DIR (contains manifest_v2.json), PAIRS_MARGIN, PAIRS_PDMS, PAIRS_NEAREST.
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
NAMES=(marginpref marginpref pdmspref pdmspref nearestpref nearestpref)
SEEDS=(20 21 20 21 20 21)
NAME_="${SCALED_MODE_NAME:-${NAMES[$TASK]}}"
SEED="${SCALED_SEED:-${SEEDS[$TASK]}}"
case "$NAME_" in
 marginpref) PAIRS="${PAIRS_MARGIN:?set PAIRS_MARGIN}" ;;
 pdmspref) PAIRS="${PAIRS_PDMS:?set PAIRS_PDMS}" ;;
 randompref) PAIRS="${PAIRS_RANDOM:?set PAIRS_RANDOM}" ;;
 nearestpref) PAIRS="${PAIRS_NEAREST:?set PAIRS_NEAREST}" ;;
 *) echo "unknown arm $NAME_"; exit 2 ;;
esac
OUTNAME="${SCALED_NAME:-scaled_v1_${NAME_}_seed${SEED}}"
ARGS=(--mode pair_pref --seed "$SEED"
 --manifest "$RUN_DIR/manifest_v2.json" --output "$RUN_DIR/$OUTNAME"
 --pairs "$PAIRS" --pref-beta "${PAIR_BETA:-200}" --winner-weight "${PAIR_WINNER_WEIGHT:-1.0}"
 --eval-set heldout --eval-scenes "${SCALED_EVAL_SCENES:-600}"
 --max-scenes 0 --epochs "${PAIR_EPOCHS:-4}" --max-updates "${SCALED_UPDATES:-800}" --lr "${SCALED_LR:-1e-5}")
PY="$HOME/.conda/envs/drivevla/bin/python"
"$PY" -m unittest discover -s tools/rl_prep -p "test_pair_preference.py"
srun "$PY" -u tools/rl_prep/train_offline_verified_policy.py "${ARGS[@]}"
