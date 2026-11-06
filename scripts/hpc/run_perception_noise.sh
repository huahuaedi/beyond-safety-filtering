#!/bin/bash
# Sensitivity of the frozen selector to synthetic perception errors (levels 1-3, see tools/rl_prep/perception_noise.py).
#   (a) "clean-trained": selector fitted on clean pool features, applied to features computed from perturbed scene state
#   (b) "matched":       selector fitted on perturbed pool features, applied to perturbed navtest features
# Applied to the navtest candidate collection of the SFT policy (seed 0); the baseline is the independent SFT evaluation.
# Usage: bash scripts/hpc/run_perception_noise.sh [levels, default "1 2 3"]
set -euo pipefail
LEVELS="${1:-1 2 3}"
OFF="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
C="$ASSET/pretrained_models/DriveVLA-W0/posttrain_runs/rl_pareto_navtest_v2_20261005_133703_37354"
O="${NOISE_COLLECTION:-$ASSET/navsim_data/exp/navtest_selector_e2e_20261009_133214_full}"
EXP="$OFF/navsim_data/exp"
SFT="$ASSET/pretrained_models/DriveVLA-W0/posttrain_runs/orig_five_navtest_20261008_223336_21405/metrics/orig_five_navtest_20261008_223336_21405_sft/2026.10.08.23.09.48/2026.10.08.23.11.33.csv"
test -d "$O/collection" && test -f "$SFT" && test -d "$EXP/world_action_geom_full_v1"
cd "$OFF"
fit() {  # fit <train-geom-root> <apply-geom-root> <output> <dependency>
  sbatch --parsable --dependency="$4" --job-name=noise_select --account=xinxia0 --partition=standard --cpus-per-task=8 --mem=64g --time=02:00:00 \
    --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
    --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $OFF/tools/rl_prep && python fit_and_apply_selector.py --train-root $EXP/world_action_full_v1 --train-geom-root $1 --apply-root $O/collection --apply-geom-root $2 --token-log-map $C/token_log_map.json --sft-csv $SFT --output $3"
}
for L in $LEVELS; do
  JN=$(DRIVEVLA_REPO_ROOT="$OFF" GEOM_TARGETS=fresh GEOM_NOISE_LEVEL=$L WM_FRESH_DATA_ROOT="$O/collection" WM_FRESH_GEOM_ROOT="$O/geom_noise$L" \
       NAVSIM_METRIC_CACHE_PATH="$C/cache" sbatch --parsable scripts/hpc/submit_extract_geometric_features.slurm)
  JP=$(DRIVEVLA_REPO_ROOT="$OFF" GEOM_TARGETS=fresh GEOM_NOISE_LEVEL=$L WM_FRESH_GEOM_ROOT="$EXP/world_action_geom_noise${L}_v1" \
       sbatch --parsable scripts/hpc/submit_extract_geometric_features.slurm)
  FA=$(fit "$EXP/world_action_geom_full_v1" "$O/geom_noise$L" "$O/selector_noise${L}_clean.json" "afterok:$JN")
  FB=$(fit "$EXP/world_action_geom_noise${L}_v1" "$O/geom_noise$L" "$O/selector_noise${L}_matched.json" "afterok:$JN:$JP")
  echo "level $L: navtest geometry $JN, pool geometry $JP, (a) clean-trained $FA, (b) matched $FB"
done
echo "results: $O/selector_noise<level>_{clean,matched}.json   (clean level 0 = $O/selector_navtest_v3.json)"
