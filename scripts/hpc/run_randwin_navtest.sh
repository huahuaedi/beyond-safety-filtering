#!/bin/bash
# Baseline: uniformly random eligible winner (pair seed 0), same loser and recipe as the standard budget.
# Tests whether any eligible winner helps, or whether geometry/proximity specifically matters.
# Usage: bash scripts/hpc/run_ablation_navtest.sh stage1 [RUN_DIR]   (CPU: build pairs; pair count must equal prefs_v3_pdms, i.e. 988)
#        bash scripts/hpc/run_ablation_navtest.sh stage2 [RUN_DIR]   (train seeds 20..23, then sharded official-navtest evaluation)
# Afterwards, per arm:  python merge_policy_eval_shards.py $X/official_navtest_trained_eval randwin_s20 $M/randwin_s20 --expect-scenes 12146
set -euo pipefail
STAGE="${1:?stage1|stage2}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
RUN="${2:-$ROOT/output/offline_scaled_20261008_150131}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
X=$ASSET/navsim_data/exp
E="$ROOT/navsim_data/exp"
DATA_REPO="${EVAL_DATA_REPO:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
NORM="$DATA_REPO/configs/normalizer_navsim_trainval/norm_stats.json"
PAIRS_DIR="$ROOT/output/prefs_v3_randwin"
cd "$ROOT"
if [ "$STAGE" = stage1 ]; then
  [ -e "$PAIRS_DIR" ] && { echo "refusing existing $PAIRS_DIR"; exit 1; }
  [ -f "$RUN/manifest_v2.json" ] || { echo "missing $RUN/manifest_v2.json"; exit 1; }
  ENVSET="module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; export PATH=\$CONDA_PREFIX/bin:\$PATH; export PYTHONPATH=$ROOT/tools/rl_prep; cd $ROOT"
  CMD="$ENVSET && python tools/rl_prep/build_margin_preferences.py --root $E/world_action_full_v1 --geom-root $E/world_action_geom_full_v1 --manifest $RUN/manifest_v2.json --norm-stats $NORM --winner-rule random --output-dir $PAIRS_DIR"
  J=$(sbatch --parsable --job-name=ablate_prep --account=xinxia0 --partition=standard --cpus-per-task=4 --mem=32g --time=02:00:00 \
      --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err --wrap "$CMD")
  echo "prep job $J -> $PAIRS_DIR  (then compare its report.json pair count with $ROOT/output/prefs_v3_pdms/report.json)"
  exit 0
fi
[ -f "$PAIRS_DIR/pairs.jsonl" ] || { echo "missing $PAIRS_DIR/pairs.jsonl"; exit 1; }
export SCALED_RUN_DIR="$RUN" PAIRS_MARGIN="$PAIRS_DIR/pairs.jsonl" PAIRS_PDMS="$PAIRS_DIR/pairs.jsonl" PAIRS_NEAREST="$PAIRS_DIR/pairs.jsonl"
OUT=$X/official_navtest_trained_eval
LIST=$OUT/arms_randwin.txt
[ -e "$LIST" ] && { echo "refusing existing $LIST"; exit 1; }
JOBS=()
for s in 20 21 22 23; do
  [ -e "$RUN/ablate_randwin_s${s}" ] && { echo "refusing existing $RUN/ablate_randwin_s${s}"; exit 1; }
  J=$(sbatch --parsable --array=0-0 --export=ALL,SCALED_MODE_NAME=pdmspref,SCALED_SEED=$s,SCALED_NAME=ablate_randwin_s${s} scripts/hpc/submit_pair_pref_train.slurm)
  echo "randwin seed $s -> job $J"; JOBS+=("$J")
  echo "randwin_s${s}|$RUN/ablate_randwin_s${s}/action_policy.pt" >> "$LIST"
done
DEP="afterok:$(IFS=:; echo "${JOBS[*]}")"
export EVAL_LIST="$LIST" EVAL_OUT="$OUT" EVAL_SHARDS=12 NT_COHORT=$X/official_navtest_full_cohort2
J=$(sbatch --parsable --array=0-47%5 --dependency="$DEP" scripts/hpc/submit_navtest_policy_eval.slurm)
echo "eval array $J (after $DEP); output $OUT"
