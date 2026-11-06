#!/bin/bash
# Selection on top of an anchor-scoring checkpoint, applied to the FULL official navtest (12,146 scenes).
# The selector is fitted on the existing anchor pool (7,171 trainval scenes of the Multi-Anchor checkpoint; its pool, geometry
# and anchor-feature directories come from an earlier run_anchor_full.sh workdir) and applied to freshly collected navtest
# candidates of the checkpoint in AN_HUB. With AN_HUB = the Query-based checkpoint this is a cross-checkpoint transfer test.
# Usage: ANCHOR_POOL_WORKDIR=<anchor_full_* dir> AN_HUB=<checkpoint dir> CHAIN_BASE_CSV=<official csv of that checkpoint> \
#        NT_ARRAY=0-47%8 bash scripts/hpc/run_anchor_navtest_full.sh <tag>
set -euo pipefail
TAG="${1:?tag, e.g. multi or query}"
OFF="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
X="$ASSET/navsim_data/exp"
COH="${NT_COHORT:-$X/official_navtest_full_cohort2}"
POOLW="${ANCHOR_POOL_WORKDIR:?set ANCHOR_POOL_WORKDIR}"
HUB="${AN_HUB:?set AN_HUB}"
CSV="${CHAIN_BASE_CSV:?set CHAIN_BASE_CSV}"
test -d "$POOLW/pool" && test -d "$POOLW/pool_geom" && test -d "$POOLW/pool_anchor_feat" && test -f "$HUB/config.json" && test -f "$COH/test.pkl"
W="$X/anchor_navtest_${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir "$W"; cd "$OFF"
JN=$(AN_HUB="$HUB" AN_TOKENS="$COH/test_tokens.json" AN_SCENE_PKL="$COH/test.pkl" AN_CACHE="$COH/cache" AN_OUT="$W/navtest" \
     sbatch --parsable --array="${NT_ARRAY:-0-47%8}" scripts/hpc/submit_anchor_candidates.slurm)
echo "navtest collection $JN   workdir $W"
GN=$(DRIVEVLA_REPO_ROOT="$OFF" GEOM_TARGETS=fresh WM_FRESH_DATA_ROOT="$W/navtest" WM_FRESH_GEOM_ROOT="$W/navtest_geom" NAVSIM_METRIC_CACHE_PATH="$COH/cache" \
     sbatch --parsable --dependency=afterok:$JN scripts/hpc/submit_extract_geometric_features.slurm)
echo "geometry $GN"
fit() {
  local out="$1"; shift
  sbatch --parsable --dependency=afterok:$GN --job-name=anchor_nt_select --account=xinxia0 --partition=standard --cpus-per-task=8 --mem=96g --time=04:00:00 \
    --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
    --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $OFF/tools/rl_prep && python fit_and_apply_selector.py --train-root $POOLW/pool --train-geom-root $POOLW/pool_geom --apply-root $W/navtest --apply-geom-root $W/navtest_geom --token-log-map $COH/token_log_map.json --sft-csv $CSV $* --output $out"
}
F1=$(fit "$W/selector_geom_only.json")
F2=$(fit "$W/selector_with_anchor_logprob.json" --train-extra-root anchor=$POOLW/pool_anchor_feat --apply-extra-root anchor=$W/navtest_anchor_feat)
echo "selectors $F1 (geometry + trajectory) and $F2 (+ anchor log-probabilities)"
echo "results: $W/selector_geom_only.json  $W/selector_with_anchor_logprob.json"
