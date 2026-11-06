#!/bin/bash
# Sensitivity of the selector to errors in the drivable-area map ONLY (rigid offset, sigma 0.2 / 0.5 / 1.0 m at levels 1-3),
# on the FULL official navtest collection; agent features stay clean. The selector is fitted on clean pool features, so this is
# the "map from a localisation estimate" situation. --split-geometry also reports trajectory+map and map-only choices.
# Usage: SEL_OUT=<full selector output dir> CHAIN_BASE_CSV=<official flow csv> bash scripts/hpc/run_map_noise_navtest.sh [levels, default "1 2 3"]
set -euo pipefail
LEVELS="${1:-1 2 3}"
OFF="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
X="$ASSET/navsim_data/exp"
O="${SEL_OUT:?set SEL_OUT}"
COH="${NT_COHORT:-$X/official_navtest_full_cohort2}"
EXP="$OFF/navsim_data/exp"
CSV="${CHAIN_BASE_CSV:?set CHAIN_BASE_CSV}"
test -d "$O/collection" && test -f "$COH/token_log_map.json" && test -d "$EXP/world_action_geom_full_v1"
cd "$OFF"
for L in $LEVELS; do
  [ -e "$O/geom_mapnoise$L" ] && { echo "refusing existing $O/geom_mapnoise$L"; exit 1; }
  JN=$(DRIVEVLA_REPO_ROOT="$OFF" GEOM_TARGETS=fresh GEOM_NOISE_LEVEL=$L GEOM_NOISE_TARGET=map WM_FRESH_DATA_ROOT="$O/collection" \
       WM_FRESH_GEOM_ROOT="$O/geom_mapnoise$L" NAVSIM_METRIC_CACHE_PATH="$COH/cache" sbatch --parsable scripts/hpc/submit_extract_geometric_features.slurm)
  FA=$(sbatch --parsable --dependency=afterok:$JN --job-name=mapnoise_select --account=xinxia0 --partition=standard --cpus-per-task=8 --mem=96g --time=03:00:00 \
       --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
       --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $OFF/tools/rl_prep && python fit_and_apply_selector.py --train-root $EXP/world_action_full_v1 --train-geom-root $EXP/world_action_geom_full_v1 --apply-root $O/collection --apply-geom-root $O/geom_mapnoise$L --token-log-map $COH/token_log_map.json --sft-csv $CSV --split-geometry --output $O/selector_mapnoise${L}.json")
  echo "level $L: geometry $JN, clean-trained selector $FA   result $O/selector_mapnoise${L}.json"
done
