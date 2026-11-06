#!/bin/bash
# Two more training seeds (22, 23) for margin and nearest-safe preference distillation (same settings as seeds 20, 21),
# then the sharded official-navtest evaluation of the four new checkpoints (12 shards each) behind them.
# Usage: bash scripts/hpc/run_seeds_navtest.sh [RUN_DIR]     RUN_DIR defaults to offline_scaled_20261008_150131.
# Afterwards, per arm:  python merge_policy_eval_shards.py $X/official_navtest_trained_eval <label> $M/<label> --expect-scenes 12146
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
RUN="${1:-$ROOT/output/offline_scaled_20261008_150131}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
X=$ASSET/navsim_data/exp
cd "$ROOT"
OUT=$X/official_navtest_trained_eval
LIST=$OUT/arms_seeds2223.txt
[ -e "$LIST" ] && { echo "refusing existing $LIST"; exit 1; }
# SKIP_TRAIN=1: the checkpoints already exist (trained earlier); only the arms whose action_policy.pt is present are evaluated.
DEP=""
if [ "${SKIP_TRAIN:-0}" != 1 ]; then
  TRAIN_LOG=$(bash scripts/hpc/submit_pair_pref_more_seeds.sh "$RUN" | tee /dev/stderr)
  DEP=$(echo "$TRAIN_LOG" | sed -n 's/^EVAL_DEP=//p')
  [ -n "$DEP" ] || { echo "no EVAL_DEP from the training submission"; exit 1; }
fi
: > "$LIST"
for a in marginpref nearestpref; do for s in 22 23; do
  CK="$RUN/scaled_v1_${a}_seed${s}/action_policy.pt"
  if [ "${SKIP_TRAIN:-0}" = 1 ] && [ ! -f "$CK" ]; then echo "missing $CK, skipped"; continue; fi
  echo "${a}_s${s}|$CK" >> "$LIST"
done; done
cat "$LIST"
N=$(wc -l < "$LIST"); [ "$N" -gt 0 ] || { echo "nothing to evaluate"; exit 1; }
export EVAL_LIST="$LIST" EVAL_OUT="$OUT" EVAL_SHARDS=12 NT_COHORT=$X/official_navtest_full_cohort2
DEPARG=(); [ -z "$DEP" ] || DEPARG=(--dependency="$DEP")
J=$(sbatch --parsable --array=0-$((N*12-1))%5 ${DEPARG[@]+"${DEPARG[@]}"} scripts/hpc/submit_navtest_policy_eval.slurm)
echo "eval array $J (after: ${DEP:-nothing}); output $OUT"
