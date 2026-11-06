#!/bin/bash
# Non-learned geometric risk features (constant-velocity clearance + drivable
# area) for every collected candidate.  CPU only; runs in the navsim env and
# shards the scenes over the array.  No GPU, no simulator rollout.
#
# TTC-aware block: GEOM_TTC=1 GEOM_TARGETS=fresh sbatch scripts/hpc/submit_extract_geometric_features.slurm
#
# Smoke test on the real metric cache before the full run:
#   GEOM_LIMIT=20 GEOM_OUT_SUFFIX=_smoke sbatch --array=0-0 scripts/hpc/submit_extract_geometric_features.slurm

#SBATCH --job-name=wm_geom_feat
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navsim
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"

EXP="$REPO_ROOT/navsim_data/exp"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$REPO_ROOT/navsim_data/maps"
export NAVSIM_EXP_ROOT="$EXP"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT/inference/navsim/navsim"
export OPENSCENE_DATA_ROOT="$REPO_ROOT/navsim_data"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CACHE="${NAVSIM_METRIC_CACHE_PATH:-$EXP/metric_cache_trainval_rl}"
LIMIT="${GEOM_LIMIT:-0}"
SUFFIX="${GEOM_OUT_SUFFIX:-}"
EXTRA=()
TTC="${GEOM_TTC:-0}"          # 1: write the TTC-aware block (world_action_ttc_*) instead of the 12 base features
if [ "$TTC" = "1" ]; then EXTRA+=(--ttc); fi
# GEOM_NOISE_LEVEL=1|2|3: synthetic perception errors on agents and map (sensitivity analysis), see tools/rl_prep/perception_noise.py
if [ "${GEOM_NOISE_LEVEL:-0}" != "0" ]; then EXTRA+=(--noise-level "$GEOM_NOISE_LEVEL" --noise-seed "${GEOM_NOISE_SEED:-0}"); fi
if [ -n "${GEOM_NOISE_TARGET:-}" ]; then EXTRA+=(--noise-target "$GEOM_NOISE_TARGET"); fi   # both | map | agents
if [ "$LIMIT" -gt 0 ]; then EXTRA+=(--limit "$LIMIT"); fi

for target in ${GEOM_TARGETS:-dev fresh}; do
    case "$target" in
        dev)   data="${WM_ACTION_DATA_ROOT:-$EXP/world_action_v1}"
               out="${WM_GEOM_ROOT:-$EXP/world_action_geom_v1}$SUFFIX" ;;
        fresh) data="${WM_FRESH_DATA_ROOT:-$EXP/world_action_full_v1}"
               if [ "$TTC" = "1" ]; then out="${WM_FRESH_TTC_ROOT:-$EXP/world_action_ttc_full_v1}$SUFFIX"
               else out="${WM_FRESH_GEOM_ROOT:-$EXP/world_action_geom_full_v1}$SUFFIX"; fi ;;
        *) echo "unknown GEOM_TARGETS entry: $target" >&2; exit 2 ;;
    esac
    echo "target=$target shard=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT data=$data cache=$CACHE out=$out limit=$LIMIT"
    python tools/rl_prep/extract_geometric_features.py \
        --data-root "$data" --metric-cache-path "$CACHE" --output-root "$out" \
        --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards "$SLURM_ARRAY_TASK_COUNT" ${EXTRA[@]+"${EXTRA[@]}"}
done
