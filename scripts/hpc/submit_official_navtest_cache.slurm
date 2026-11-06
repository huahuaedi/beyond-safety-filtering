#!/bin/bash
#SBATCH --job-name=navtest_cache
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=10:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
# Official NAVSIM metric cache for the FULL navtest split (147 logs, ~12k scenes), built with the shipped
# run_metric_caching.py and the stock `navtest` scene filter. CPU only, one time; every method is later scored against it.
# Output: $E/exp/metric_cache_navtest_official  (E = the DriveVLA-W0-data/navsim_data root)
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
E="${NAVSIM_DATA:-/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data/navsim_data}"
OUT="$E/exp/metric_cache_navtest_official"
[ -e "$OUT" ] && { echo "refusing existing $OUT"; exit 1; }

# The stock config reads $OPENSCENE_DATA_ROOT/navsim_logs/test/*.pkl; here the logs sit one level deeper (test/test).
LOGS="$E/navsim_logs/test"
ls "$LOGS"/*.pkl >/dev/null 2>&1 || LOGS="$LOGS/test"
N=$(ls "$LOGS"/*.pkl 2>/dev/null | wc -l)
echo "navtest log pickles at $LOGS: $N"
[ "$N" -eq 147 ] || { echo "expected 147 navtest logs, found $N"; exit 2; }
VIEW="$HOME/openscene_navtest_view"
mkdir -p "$VIEW/navsim_logs" "$VIEW/sensor_blobs"
ln -sfn "$LOGS" "$VIEW/navsim_logs/test"
ln -sfn "$E/sensor_blobs" "$VIEW/sensor_blobs/test_parent"   # not needed for metric caching; kept for completeness

module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navsim
export PATH="$CONDA_PREFIX/bin:$PATH"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="$E/maps"
export OPENSCENE_DATA_ROOT="$VIEW"
export NAVSIM_EXP_ROOT="$E/exp"
export NAVSIM_DEVKIT_ROOT="$ROOT/inference/navsim/navsim"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest \
  cache.cache_path="$OUT"
echo "cached scenes: $(find "$OUT" -name 'metric_cache.pkl' | wc -l)"
echo NAVTEST_CACHE_DONE
