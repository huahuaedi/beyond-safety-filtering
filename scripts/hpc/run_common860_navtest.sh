#!/bin/bash
# Common-scene control: rule RULE (nearest|pdms) trained only on the scenes that have a pdms_capped pair (860 of 988), so that
# capped vs. uncapped/nearest comparisons do not differ in training scenes. Same loser, recipe, seeds 20..23.
# Usage: bash scripts/hpc/run_common860_navtest.sh RULE stage1 [RUN_DIR]   (CPU: build pairs; count must be 860)
#        bash scripts/hpc/run_common860_navtest.sh RULE stage2 [RUN_DIR]   (train seeds 20..23, then sharded official-navtest evaluation)
# Afterwards, per arm:  python merge_policy_eval_shards.py $X/official_navtest_trained_eval ${RULE}c860_s20 $M/${RULE}c860_s20 --expect-scenes 12146
set -euo pipefail
RULE="${1:?nearest|pdms}"; STAGE="${2:?stage1|stage2}"; case "$RULE" in nearest|pdms) ;; *) echo bad rule; exit 1;; esac
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
RUN="${3:-$ROOT/output/offline_scaled_20261008_150131}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
X=$ASSET/navsim_data/exp
E="$ROOT/navsim_data/exp"
DATA_REPO="${EVAL_DATA_REPO:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
NORM="$DATA_REPO/configs/normalizer_navsim_trainval/norm_stats.json"
CAPPED="$ROOT/output/prefs_v3_pdmscapped/pairs.jsonl"
PAIRS_DIR="$ROOT/output/prefs_v3_${RULE}_c860"
cd "$ROOT"
if [ "$STAGE" = stage1 ]; then
  [ -e "$PAIRS_DIR" ] && { echo "refusing existing $PAIRS_DIR"; exit 1; }
  [ -f "$RUN/manifest_v2.json" ] || { echo "missing $RUN/manifest_v2.json"; exit 1; }
  ENVSET="module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; export PATH=\$CONDA_PREFIX/bin:\$PATH; export PYTHONPATH=$ROOT/tools/rl_prep; cd $ROOT"
  CMD="$ENVSET && python tools/rl_prep/build_margin_preferences.py --root $E/world_action_full_v1 --geom-root $E/world_action_geom_full_v1 --manifest $RUN/manifest_v2.json --norm-stats $NORM --winner-rule $RULE --restrict-to $CAPPED --output-dir $PAIRS_DIR"
  J=$(sbatch --parsable --job-name=c860_prep --account=xinxia0 --partition=standard --cpus-per-task=4 --mem=32g --time=02:00:00 \
      --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err --wrap "$CMD")
  echo "prep job $J -> $PAIRS_DIR  (pair count must be 860)"
  exit 0
fi
[ -f "$PAIRS_DIR/pairs.jsonl" ] || { echo "missing $PAIRS_DIR/pairs.jsonl"; exit 1; }
export SCALED_RUN_DIR="$RUN" PAIRS_MARGIN="$PAIRS_DIR/pairs.jsonl" PAIRS_PDMS="$PAIRS_DIR/pairs.jsonl" PAIRS_NEAREST="$PAIRS_DIR/pairs.jsonl"
OUT=$X/official_navtest_trained_eval
LIST=$OUT/arms_${RULE}_c860.txt
[ -e "$LIST" ] && { echo "refusing existing $LIST"; exit 1; }
JOBS=()
for s in 20 21 22 23; do
  [ -e "$RUN/c860_${RULE}_s${s}" ] && { echo "refusing existing $RUN/c860_${RULE}_s${s}"; exit 1; }
  J=$(sbatch --parsable --array=0-0 --export=ALL,SCALED_MODE_NAME=${RULE}pref,SCALED_SEED=$s,SCALED_NAME=c860_${RULE}_s${s} scripts/hpc/submit_pair_pref_train.slurm)
  echo "c860 ${RULE} seed $s -> job $J"; JOBS+=("$J")
  echo "${RULE}c860_s${s}|$RUN/c860_${RULE}_s${s}/action_policy.pt" >> "$LIST"
done
DEP="afterok:$(IFS=:; echo "${JOBS[*]}")"
export EVAL_LIST="$LIST" EVAL_OUT="$OUT" EVAL_SHARDS=12 NT_COHORT=$X/official_navtest_full_cohort2
J=$(sbatch --parsable --array=0-47%5 --dependency="$DEP" scripts/hpc/submit_navtest_policy_eval.slurm)
echo "eval array $J (after $DEP); output $OUT"
