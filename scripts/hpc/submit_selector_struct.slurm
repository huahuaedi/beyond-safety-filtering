#!/bin/bash
# Component-wise structured scorer (and optionally set-level features) against the current boosted scorer, on the log-disjoint
# candidate data. CPU only. Development happens here; navtest is touched once, at the end, with the frozen choice.
# Env: SELECTOR_MANIFEST (manifest_v2.json), SET_FEATURES=0|1, STRUCT_OUT (output json).
#SBATCH --job-name=selector_struct
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
set -euo pipefail
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"
EXP="$REPO_ROOT/navsim_data/exp"
MANIFEST="${SELECTOR_MANIFEST:?SELECTOR_MANIFEST}"
OUT="${STRUCT_OUT:?STRUCT_OUT}"
TTC_ROOT="${WM_FRESH_TTC_ROOT:-$EXP/world_action_ttc_full_v1}"
FLAGS=()
[ "${SET_FEATURES:-0}" = "1" ] && FLAGS+=(--set-features)
export OMP_NUM_THREADS=8
python tools/rl_prep/analyze_trajectory_selector.py \
    --root "${WM_FRESH_DATA_ROOT:-$EXP/world_action_full_v1}" \
    --geom-root "${WM_FRESH_GEOM_ROOT:-$EXP/world_action_geom_full_v1}" \
    --manifest "$MANIFEST" --output "$OUT" \
    --extra-root "ttc=$TTC_ROOT" \
    --metric-cache-dirs "${NAVSIM_METRIC_CACHE_PATH:-$EXP/metric_cache_trainval_rl}" \
    --hgb --struct --hgb-sets "traj+geom,traj+geom+ttc" "${FLAGS[@]}"
