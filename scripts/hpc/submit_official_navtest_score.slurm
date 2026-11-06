#!/bin/bash
#SBATCH --job-name=nt_official_score
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
# Official PDM scoring (run_pdm_score.py + emu_vla_agent) of a directory of per-scene json actions on the FULL navtest cache.
# Env: SCORE_JSON (dir with <token>.json), SCORE_NAME (experiment name), ANCHOR_BASED (true for the anchor-scoring planner, else false).
set -euo pipefail
JSON="${SCORE_JSON:?SCORE_JSON}"; NAME="${SCORE_NAME:?SCORE_NAME}"; ANCHOR="${ANCHOR_BASED:-false}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
E=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data/navsim_data
CACHE="${NAVTEST_CACHE:-$E/exp/metric_cache_navtest_official}"
VIEW="$HOME/openscene_navtest_view"
test -d "$CACHE" && test -d "$JSON" && test -d "$VIEW/navsim_logs/test"
echo "json files: $(ls "$JSON"/*.json | wc -l)"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navsim
export PATH="$CONDA_PREFIX/bin:$PATH"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT="$E/maps"
export OPENSCENE_DATA_ROOT="$VIEW" NAVSIM_EXP_ROOT="$E/exp"
export NAVSIM_DEVKIT_ROOT="$ROOT/inference/navsim/navsim"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split=navtest agent=emu_vla_agent agent.experiment_path="$JSON" agent.anchor_based="$ANCHOR" \
  experiment_name="$NAME" metric_cache_path="$CACHE"
echo SCORE_DONE
