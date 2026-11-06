#!/bin/bash
# Larger-budget PDMS-winner control: reuses the EXISTING scale-up run dir (same manifest_v2.json, hence the same training
# scenes as scaleup_marginpref_s20 / scaleup_nearestpref_s20), builds PDMS pairs there, trains with 2x updates, evaluates.
# Usage: bash scripts/hpc/run_scaleup_pdms.sh stage1 <EXISTING_SCALEUP_RUN_DIR>   (CPU: PDMS pairs; check the pair count)
#        bash scripts/hpc/run_scaleup_pdms.sh stage2 <EXISTING_SCALEUP_RUN_DIR>   (training + sharded navtest evaluation)
set -euo pipefail
STAGE="${1:?stage1|stage2}"
RUN="${2:?existing scale-up run dir containing manifest_v2.json}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
X=$ASSET/navsim_data/exp
E="$ROOT/navsim_data/exp"
DATA_REPO="${EVAL_DATA_REPO:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
NORM="$DATA_REPO/configs/normalizer_navsim_trainval/norm_stats.json"
cd "$ROOT"
[ -f "$RUN/manifest_v2.json" ] || { echo "missing $RUN/manifest_v2.json"; exit 1; }
if [ "$STAGE" = stage1 ]; then
  [ -e "$RUN/prefs_pdms" ] && { echo "refusing existing $RUN/prefs_pdms"; exit 1; }
  ENVSET="module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; export PATH=\$CONDA_PREFIX/bin:\$PATH; export PYTHONPATH=$ROOT/tools/rl_prep; cd $ROOT"
  CMD="$ENVSET && python tools/rl_prep/build_margin_preferences.py --root $E/world_action_full_v1 --geom-root $E/world_action_geom_full_v1 --manifest $RUN/manifest_v2.json --norm-stats $NORM --winner-rule pdms --output-dir $RUN/prefs_pdms"
  J=$(sbatch --parsable --job-name=scaleup_pdms_prep --account=xinxia0 --partition=standard --cpus-per-task=4 --mem=32g --time=01:00:00 \
      --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err --wrap "$CMD")
  echo "prep job $J (then: bash scripts/hpc/run_scaleup_pdms.sh stage2 $RUN)"
  exit 0
fi
[ -f "$RUN/prefs_pdms/pairs.jsonl" ] || { echo "missing $RUN/prefs_pdms/pairs.jsonl"; exit 1; }
export SCALED_RUN_DIR="$RUN" PAIRS_MARGIN="$RUN/prefs_margin/pairs.jsonl" PAIRS_NEAREST="$RUN/prefs_nearest/pairs.jsonl" PAIRS_PDMS="$RUN/prefs_pdms/pairs.jsonl"
J=$(sbatch --parsable --array=0-0 --time=08:00:00 --export=ALL,SCALED_MODE_NAME=pdmspref,SCALED_SEED=20,SCALED_NAME=scaleup_pdmspref_s20,SCALED_UPDATES=1600,SCALED_EVAL_SCENES=100 scripts/hpc/submit_pair_pref_train.slurm)
echo "pdmspref -> $J"
OUT=$X/official_navtest_trained_eval
LIST=$OUT/arms_scaleup_pdms.txt
[ -e "$LIST" ] && { echo "refusing existing $LIST"; exit 1; }
echo "scaleup_pdmspref_s20|$RUN/scaleup_pdmspref_s20/action_policy.pt" > "$LIST"
export EVAL_LIST="$LIST" EVAL_OUT="$OUT" EVAL_SHARDS=12 NT_COHORT=$X/official_navtest_full_cohort2
E2=$(sbatch --parsable --array=0-11%5 --dependency="afterok:$J" scripts/hpc/submit_navtest_policy_eval.slurm)
echo "eval array $E2 (after $J); output $OUT"
