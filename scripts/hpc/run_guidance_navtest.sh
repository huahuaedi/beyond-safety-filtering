#!/bin/bash
# One-shot test of cost-guided sampling on the frozen navtest cohort, paired with the unguided seed-0 collection.
# Pre-registered: primary lambda = 0.4 (best on the development hard scenes), secondary lambda = 0.2. Nothing is tuned here.
# Usage: bash scripts/hpc/run_guidance_navtest.sh [lambdas, default "0.4 0.2"]
set -euo pipefail
LAMBDAS="${1:-0.4 0.2}"
OFF="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
C="$ASSET/pretrained_models/DriveVLA-W0/posttrain_runs/rl_pareto_navtest_v2_20261005_133703_37354"
EXP="$OFF/navsim_data/exp"
SFT="$ASSET/pretrained_models/DriveVLA-W0/posttrain_runs/orig_five_navtest_20261008_223336_21405/metrics/orig_five_navtest_20261008_223336_21405_sft/2026.10.08.23.09.48/2026.10.08.23.11.33.csv"
test -f "$SFT" && test -f "$C/test.pkl" && test -d "$C/cache"
W="$ASSET/navsim_data/exp/guidance_navtest_$(date +%Y%m%d_%H%M%S)"
mkdir "$W"; cd "$OFF"
JE=$(GUIDE_TOKENS="$C/test_tokens.json" GUIDE_CACHE="$C/cache" GUIDE_OUT="$W/guide_inputs" DRIVEVLA_REPO_ROOT="$OFF" \
     sbatch --parsable --array=0-7 scripts/hpc/submit_extract_guidance_inputs.slurm)
echo "guidance inputs job $JE   (workdir $W)"
for L in $LAMBDAS; do
  OUT="$W/lam_$L"; mkdir -p "$OUT"
  JC=$(GD_OUT="$OUT" GD_TOKENS="$C/test_tokens.json" GD_SCENE_PKL="$C/test.pkl" GD_CACHE="$C/cache" GD_GUIDE_DIR="$W/guide_inputs" GD_LAMBDA="$L" RL_SEED=0 \
       sbatch --parsable --array=0-3 --dependency=afterok:$JE scripts/hpc/submit_guided_candidates.slurm)
  JG=$(DRIVEVLA_REPO_ROOT="$OFF" GEOM_TARGETS=fresh WM_FRESH_DATA_ROOT="$OUT/collection" WM_FRESH_GEOM_ROOT="$OUT/geom" NAVSIM_METRIC_CACHE_PATH="$C/cache" \
       sbatch --parsable --dependency=afterok:$JC scripts/hpc/submit_extract_geometric_features.slurm)
  JF=$(sbatch --parsable --dependency=afterok:$JG --job-name=guide_nt_select --account=xinxia0 --partition=standard --cpus-per-task=8 --mem=64g --time=02:00:00 \
       --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
       --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $OFF/tools/rl_prep && python fit_and_apply_selector.py --train-root $EXP/world_action_full_v1 --train-geom-root $EXP/world_action_geom_full_v1 --apply-root $OUT/collection --apply-geom-root $OUT/geom --token-log-map $C/token_log_map.json --sft-csv $SFT --output $OUT/selector_navtest.json")
  echo "lambda $L: collection $JC, geometry $JG, selector $JF -> $OUT/selector_navtest.json"
done
echo "unguided control (seed 0): $ASSET/navsim_data/exp/navtest_selector_e2e_20261009_133214_full/selector_navtest_v3.json"
