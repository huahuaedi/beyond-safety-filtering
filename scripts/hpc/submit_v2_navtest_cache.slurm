#!/bin/bash
#SBATCH --job-name=v2_cache
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=10:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
# NAVSIM v2 metric cache for navtest (12,146 tokens of the v2 `navtest` filter), same logs/maps as the v1 cache.
# SMOKE=1: 30 scenes into metric_cache_navtest_v2_smoke (check this first).
set -euo pipefail
V2="${NAVSIM_V2_REPO:-$HOME/navsim_v2}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
E=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data/navsim_data
VIEW="$HOME/openscene_navtest_view"
test -d "$VIEW/navsim_logs/test"
OUT="$E/exp/metric_cache_navtest_v2"; EXTRA=()
if [ "${SMOKE:-0}" = 1 ]; then OUT="${OUT}_smoke"; EXTRA=(train_test_split.scene_filter.max_scenes=30); fi
[ -e "$OUT" ] && { echo "refusing existing $OUT"; exit 1; }
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navsim2
export PATH="$CONDA_PREFIX/bin:$PATH"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT="$E/maps"
export OPENSCENE_DATA_ROOT="$VIEW" NAVSIM_EXP_ROOT="$E/exp" NAVSIM_DEVKIT_ROOT="$V2"
export PYTHONPATH="$V2:${PYTHONPATH:-}"
cd "$V2"
python navsim/planning/script/run_metric_caching.py train_test_split=navtest metric_cache_path="$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
echo "cached scenes: $(find "$OUT" -name 'metric_cache.pkl' | wc -l)"
echo V2_CACHE_DONE
