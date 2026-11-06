#!/bin/bash
# Selection on top of the anchor-scoring planner (Emu3_Multi_Anchor_PDMS_90.3): top-16 anchors per scene.
#   pool (7171 scenes) -> train the selector;  navtest cohort (1000 scenes, 136 logs) -> evaluate it.
# Candidate 0 is the planner's own output, so every gain is a gain over that planner scored by the same worker.
# Two selectors are fitted on the pool: with and without the planner's anchor log-probabilities as features.
# Usage: bash scripts/hpc/run_anchor_full.sh
set -euo pipefail
OFF="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
C="$ASSET/pretrained_models/DriveVLA-W0/posttrain_runs/rl_pareto_navtest_v2_20261005_133703_37354"
POOL_CACHE="$OFF/navsim_data/exp/metric_cache_trainval_rl"
POOL_PKL="$ASSET/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
POOL_TOKENS="$ASSET/data/navsim/processed_data/anchor/trainval_rl_scene_tokens_cached.json"
test -f "$POOL_TOKENS" && test -f "$C/test.pkl" && test -d "$POOL_CACHE"
W="$ASSET/navsim_data/exp/anchor_full_$(date +%Y%m%d_%H%M%S)"
mkdir "$W"; cd "$OFF"
JP=$(AN_TOKENS="$POOL_TOKENS" AN_SCENE_PKL="$POOL_PKL" AN_CACHE="$POOL_CACHE" AN_OUT="$W/pool" sbatch --parsable --array=0-7 scripts/hpc/submit_anchor_candidates.slurm)
JN=$(AN_TOKENS="$C/test_tokens.json" AN_SCENE_PKL="$C/test.pkl" AN_CACHE="$C/cache" AN_OUT="$W/navtest" sbatch --parsable --array=0-3 scripts/hpc/submit_anchor_candidates.slurm)
echo "pool collection $JP (8 shards), navtest collection $JN (4 shards)   workdir $W"
GP=$(DRIVEVLA_REPO_ROOT="$OFF" GEOM_TARGETS=fresh WM_FRESH_DATA_ROOT="$W/pool" WM_FRESH_GEOM_ROOT="$W/pool_geom" NAVSIM_METRIC_CACHE_PATH="$POOL_CACHE" \
     sbatch --parsable --dependency=afterok:$JP scripts/hpc/submit_extract_geometric_features.slurm)
GN=$(DRIVEVLA_REPO_ROOT="$OFF" GEOM_TARGETS=fresh WM_FRESH_DATA_ROOT="$W/navtest" WM_FRESH_GEOM_ROOT="$W/navtest_geom" NAVSIM_METRIC_CACHE_PATH="$C/cache" \
     sbatch --parsable --dependency=afterok:$JN scripts/hpc/submit_extract_geometric_features.slurm)
echo "geometry: pool $GP, navtest $GN"
fit() {  # fit <output> [extra args]
  local out="$1"; shift
  sbatch --parsable --dependency=afterok:$GP:$GN --job-name=anchor_select --account=xinxia0 --partition=standard --cpus-per-task=8 --mem=96g --time=03:00:00 \
    --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
    --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $OFF/tools/rl_prep && python fit_and_apply_selector.py --train-root $W/pool --train-geom-root $W/pool_geom --apply-root $W/navtest --apply-geom-root $W/navtest_geom --token-log-map $C/token_log_map.json $* --output $out"
}
F1=$(fit "$W/selector_geom_only.json")
F2=$(fit "$W/selector_with_anchor_logprob.json" --train-extra-root anchor=${W}/pool_anchor_feat --apply-extra-root anchor=${W}/navtest_anchor_feat)
echo "selectors: without anchor features $F1, with anchor log-probabilities $F2"
echo "results: $W/selector_geom_only.json, $W/selector_with_anchor_logprob.json"
