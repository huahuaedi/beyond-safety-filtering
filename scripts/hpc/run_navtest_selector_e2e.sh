#!/bin/bash
# Usage: bash scripts/hpc/run_navtest_selector_e2e.sh smoke|full   (run on the HPC login node)
#        CHAIN_OUT=<full output dir> CHAIN_JOB=<collection array job id> bash scripts/hpc/run_navtest_selector_e2e.sh chain
#          (queues geometry + fit-and-apply behind a collection array that was submitted by hand)
#   smoke: collect 2 navtest scenes with 16 candidates, then verify the output format. No selector yet.
#   full : collect all 1000 scenes (4 shards) -> geometric features (CPU) -> fit selector on the pool and apply it.
# Needs: DriveVLA-W0 (collection code and checkpoints) and DriveVLA-Safety-RL-offline-20260907 (analysis code and pool data).
set -euo pipefail
STAGE="${1:?smoke|full}"
WORK="${DRIVEVLA_REPO_ROOT:?set DRIVEVLA_REPO_ROOT to a recent repo copy that has RL_TRAIN_TOKENS support, e.g. ~/rl_safety_order_v1_20261008_150538_33627/repo (NOT ~/DriveVLA-W0, which is from August)}"
OFFLINE="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
# NT_COHORT: another cohort directory (test.pkl, test_tokens.json, cache/, token_log_map.json), e.g. the full official navtest
# built by tools/rl_prep/prepare_official_cohort.py. NT_ARRAY: collection array override, e.g. 0-47%8 for 12146 scenes.
COHORT="${NT_COHORT:-$ASSET/pretrained_models/DriveVLA-W0/posttrain_runs/rl_pareto_navtest_v2_20261005_133703_37354}"
EXP="$OFFLINE/navsim_data/exp"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ASSET/navsim_data/exp/navtest_selector_e2e_${STAMP}_${STAGE}"
if [ "$STAGE" != "chain" ]; then
  [ -e "$OUT" ] && { echo "refusing existing $OUT"; exit 1; }
  mkdir -p "$OUT"
fi
export NT_OUT="$OUT" NT_COHORT="$COHORT" DRIVEVLA_REPO_ROOT="$WORK"

if [ "$STAGE" = "smoke" ]; then
  J=$(NT_STEPS=2 sbatch --parsable --array=0-0 --time=01:30:00 "$OFFLINE/scripts/hpc/submit_navtest_candidates.slurm")
  echo "smoke collection job $J   output $OUT"
  echo "afterwards: python3 $OFFLINE/tools/rl_prep/check_navtest_collection.py $OUT/collection"
  exit 0
fi

chain() {  # chain <OUT> <collection-job-id>: geometry features, then fit on the pool and apply to the collection
  local OUT="$1" J1="$2"
  # geometric features reuse the existing CPU job; it reads GEOM_* / WM_FRESH_* / NAVSIM_METRIC_CACHE_PATH from the environment
  local J2 J3 SFT_CSV
  J2=$(DRIVEVLA_REPO_ROOT="$OFFLINE" GEOM_TARGETS=fresh WM_FRESH_DATA_ROOT="$OUT/collection" WM_FRESH_GEOM_ROOT="$OUT/geom" \
       NAVSIM_METRIC_CACHE_PATH="$COHORT/cache" sbatch --parsable --dependency=afterok:$J1 "$OFFLINE/scripts/hpc/submit_extract_geometric_features.slurm")
  echo "geometry array $J2"
  SFT_CSV="${CHAIN_BASE_CSV:-}"
  [ -n "$SFT_CSV" ] || SFT_CSV=$(ls "$ASSET"/pretrained_models/DriveVLA-W0/posttrain_runs/orig_five_navtest_*/metrics/*_sft/*/*.csv 2>/dev/null | grep -v smoke | tail -1 || true)
  J3=$(sbatch --parsable --dependency=afterok:$J2 --job-name=nt_select --account=xinxia0 --partition=standard --cpus-per-task=8 --mem=48g --time=02:00:00 \
       --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
       --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $OFFLINE/tools/rl_prep && python fit_and_apply_selector.py --train-root $EXP/world_action_full_v1 --train-geom-root $EXP/world_action_geom_full_v1 --apply-root $OUT/collection --apply-geom-root $OUT/geom --token-log-map $COHORT/token_log_map.json ${SFT_CSV:+--sft-csv $SFT_CSV} --model-out $OUT/selector_models.pkl --output $OUT/selector_navtest.json")
  echo "fit-and-apply job $J3   result: $OUT/selector_navtest.json"
}

if [ "$STAGE" = "chain" ]; then
  chain "${CHAIN_OUT:?set CHAIN_OUT to the full collection output dir}" "${CHAIN_JOB:?set CHAIN_JOB to the collection array job id}"
  exit 0
fi

J1=$(sbatch --parsable ${NT_ARRAY:+--array=$NT_ARRAY} "$OFFLINE/scripts/hpc/submit_navtest_candidates.slurm")
echo "collection array $J1   output $OUT"
chain "$OUT" "$J1"
