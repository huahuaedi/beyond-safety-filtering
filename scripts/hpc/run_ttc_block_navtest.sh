#!/bin/bash
# TTC-aware feature block on the FULL official navtest collection, then the selector with and without it.
#   Needs: a finished full collection (navtest_selector_e2e_*_full) and the pool's TTC block (world_action_ttc_full_v1).
# Usage: SEL_OUT=<full output dir> CHAIN_BASE_CSV=<official flow csv> bash scripts/hpc/run_ttc_block_navtest.sh
set -euo pipefail
OFF="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
X="$ASSET/navsim_data/exp"
O="${SEL_OUT:?set SEL_OUT to the full selector output directory}"
COH="${NT_COHORT:-$X/official_navtest_full_cohort2}"
EXP="$OFF/navsim_data/exp"
CSV="${CHAIN_BASE_CSV:?set CHAIN_BASE_CSV to the official SFT per-scene csv}"
test -d "$O/collection" && test -d "$EXP/world_action_ttc_full_v1" && test -f "$COH/token_log_map.json"
[ -e "$O/ttc" ] && { echo "refusing existing $O/ttc"; exit 1; }
cd "$OFF"
JT=$(DRIVEVLA_REPO_ROOT="$OFF" GEOM_TTC=1 GEOM_TARGETS=fresh WM_FRESH_DATA_ROOT="$O/collection" WM_FRESH_TTC_ROOT="$O/ttc" \
     NAVSIM_METRIC_CACHE_PATH="$COH/cache" sbatch --parsable scripts/hpc/submit_extract_geometric_features.slurm)
echo "ttc extraction array $JT"
JF=$(sbatch --parsable --dependency=afterok:$JT --job-name=nt_ttc_select --account=xinxia0 --partition=standard --cpus-per-task=8 --mem=96g --time=04:00:00 \
     --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
     --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $OFF/tools/rl_prep && python fit_and_apply_selector.py --train-root $EXP/world_action_full_v1 --train-geom-root $EXP/world_action_geom_full_v1 --train-extra-root ttc=$EXP/world_action_ttc_full_v1 --apply-root $O/collection --apply-geom-root $O/geom --apply-extra-root ttc=$O/ttc --token-log-map $COH/token_log_map.json --sft-csv $CSV --output $O/selector_navtest_ttc.json")
echo "fit-and-apply $JF   result $O/selector_navtest_ttc.json"
