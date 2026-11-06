#!/bin/bash
# Score an existing directory of 286 per-token prediction JSONs without rerunning VLA inference.

#SBATCH --job-name=score_subset
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail
: "${SCORE_EXPERIMENT_PATH:?set SCORE_EXPERIMENT_PATH to the existing per-token JSON directory}"
: "${SCORE_EXPERIMENT_NAME:?set SCORE_EXPERIMENT_NAME to a unique result name}"

REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navsim
cd "$REPO_ROOT"

NAVSIM_PYTHON="${NAVSIM_PYTHON:-$HOME/.conda/envs/navsim/bin/python}"
if [[ ! -x "$NAVSIM_PYTHON" ]]; then
  echo "navsim Python is not executable: $NAVSIM_PYTHON" >&2
  exit 1
fi
echo "navsim_python=$NAVSIM_PYTHON"
"$NAVSIM_PYTHON" -c 'import hydra; print("hydra_import=OK")'

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$REPO_ROOT/navsim_data/maps"
export NAVSIM_EXP_ROOT="$REPO_ROOT/navsim_data/exp"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT/inference/navsim/navsim"
export OPENSCENE_DATA_ROOT="$REPO_ROOT/navsim_data"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

"$NAVSIM_PYTHON" tools/rl_prep/eval_pdm_score_subset.py \
  --experiment-path "$SCORE_EXPERIMENT_PATH" \
  --experiment-name "$SCORE_EXPERIMENT_NAME" \
  --metric-cache-path "$NAVSIM_EXP_ROOT/metric_cache_trainval_rl_eval" \
  --subset-tokens-json "$REPO_ROOT/data/navsim/processed_data/anchor/trainval_rl_eval_tokens_cached.json" \
  --train-test-split trainval
