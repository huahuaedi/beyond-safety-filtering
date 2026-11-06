#!/bin/bash
# Selector with the TTC block and/or world-model PCA block on the log-disjoint split (CPU).
# Pre-registered comparison on the held-out logs (HGB, same hyper-parameter search for every set):
#   traj+geom  vs  traj+geom+ttc            -> does the TTC block close the TTC gap?
#   traj+geom  vs  traj+geom+wm             -> does the world model add information? (R2)
#   traj+geom+wm vs traj+geom+wm_shuf       -> noise control with the same dimension
# The world-model layer is fixed in advance (WM_LAYER, default 0 = layer -1); other layers are
# run only as a labelled secondary analysis, never selected by held-out score.

#SBATCH --job-name=selector_extras
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"

EXP="$REPO_ROOT/navsim_data/exp"
MANIFEST="${SELECTOR_MANIFEST:?SELECTOR_MANIFEST}"
LAYER="${WM_LAYER:-0}"
TTC_ROOT="${WM_FRESH_TTC_ROOT:-$EXP/world_action_ttc_full_v1}"
PCA_ROOT="${WM_PCA_ROOT:-$EXP/world_action_wm_pca_full_v1}/L$LAYER"
OUT="${SELECTOR_EXTRAS_OUT:-$REPO_ROOT/output/selector_extras_L$LAYER.json}"
# SELECTOR_SKIP_WM=1: TTC block only (no world-model block, no wm sets)
SKIP_WM="${SELECTOR_SKIP_WM:-0}"
BLOCKS=("$TTC_ROOT"); [ "$SKIP_WM" = "1" ] || BLOCKS+=("$PCA_ROOT")
for d in "${BLOCKS[@]}"; do [ -d "$d/train" ] || { echo "missing feature block: $d" >&2; exit 2; }; done
if [ "$SKIP_WM" = "1" ]; then
    EXTRA_ARGS=(--extra-root "ttc=$TTC_ROOT" --hgb-sets "traj+geom,traj+geom+ttc")
    OUT="${SELECTOR_EXTRAS_OUT:-$REPO_ROOT/output/selector_extras_ttc_only.json}"
else
    EXTRA_ARGS=(--extra-root "ttc=$TTC_ROOT" --extra-root "wm=$PCA_ROOT" --shuffle-control wm
                --hgb-sets "traj+geom,traj+geom+ttc,traj+geom+wm,traj+geom+ttc+wm,traj+geom+wm_shuf")
fi

export OMP_NUM_THREADS=8
python tools/rl_prep/analyze_trajectory_selector.py \
    --root "${WM_FRESH_DATA_ROOT:-$EXP/world_action_full_v1}" \
    --geom-root "${WM_FRESH_GEOM_ROOT:-$EXP/world_action_geom_full_v1}" \
    --manifest "$MANIFEST" --output "$OUT" \
    --metric-cache-dirs "${NAVSIM_METRIC_CACHE_PATH:-$EXP/metric_cache_trainval_rl}" \
    --hgb "${EXTRA_ARGS[@]}"
