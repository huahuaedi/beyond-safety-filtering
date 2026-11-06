#!/bin/bash
# Per-scene inputs of the geometric guidance cost (agents at t=0 and a drivable-area signed-distance raster in the ego frame).
# CPU only, navsim env. Env: GUIDE_TOKENS (json list of scene tokens), GUIDE_CACHE (metric cache dir), GUIDE_OUT (output dir),
# optional GUIDE_LIMIT. Smoke: GUIDE_LIMIT=20 sbatch --array=0-0 scripts/hpc/submit_extract_guidance_inputs.slurm
#SBATCH --job-name=guide_inputs
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
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navsim
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"
EXP="$REPO_ROOT/navsim_data/exp"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" NUPLAN_MAPS_ROOT="$REPO_ROOT/navsim_data/maps"
export NAVSIM_EXP_ROOT="$EXP" NAVSIM_DEVKIT_ROOT="$REPO_ROOT/inference/navsim/navsim" OPENSCENE_DATA_ROOT="$REPO_ROOT/navsim_data"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EXTRA=()
[ "${GUIDE_LIMIT:-0}" -gt 0 ] && EXTRA+=(--limit "$GUIDE_LIMIT")
python tools/rl_prep/extract_guidance_inputs.py \
    --tokens-json "${GUIDE_TOKENS:?GUIDE_TOKENS}" --metric-cache-path "${GUIDE_CACHE:?GUIDE_CACHE}" --output-root "${GUIDE_OUT:?GUIDE_OUT}" \
    --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards "$SLURM_ARRAY_TASK_COUNT" ${EXTRA[@]+"${EXTRA[@]}"}
