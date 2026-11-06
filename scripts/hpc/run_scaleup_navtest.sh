#!/bin/bash
# Scale-up of geometry-label preference distillation: train on all log-disjoint pool scenes except a small held-out slice
# (about 2x the 3,481 scenes of the earlier arms), 2x the update budget, then sharded official-navtest evaluation.
# Same recipe otherwise (beta, lr, winner weight, pass rule). Labels: margin and nearest-safe, seed 20.
# Usage: bash scripts/hpc/run_scaleup_navtest.sh stage1   (CPU: manifest + pairs; check the printed scene counts and the pair counts)
#        bash scripts/hpc/run_scaleup_navtest.sh stage2 <RUN_DIR>   (training and evaluation, RUN_DIR printed by stage1)
set -euo pipefail
STAGE="${1:?stage1|stage2}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
X=$ASSET/navsim_data/exp
E="$ROOT/navsim_data/exp"
DATA_REPO="${EVAL_DATA_REPO:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
NORM="$DATA_REPO/configs/normalizer_navsim_trainval/norm_stats.json"
cd "$ROOT"
ENVSET="module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; export PATH=\$CONDA_PREFIX/bin:\$PATH; export PYTHONPATH=$ROOT/tools/rl_prep; cd $ROOT"
if [ "$STAGE" = stage1 ]; then
  RUN="$ROOT/output/offline_scaleup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$RUN"
  echo '[]' > "$RUN/empty_tokens.json"
  CMD="$ENVSET && python tools/rl_prep/build_log_disjoint_manifest.py --root $E/world_action_full_v1 --metric-cache-dirs $E/metric_cache_trainval_rl $E/metric_cache_trainval_rl_eval $E/metric_cache_checkpoint_selection_val --final-eval-tokens $RUN/empty_tokens.json --val-tokens $RUN/empty_tokens.json --output $RUN/manifest_v2.json --label-mode topk_safe --topk 2 --extra-heldout-log-fraction 0.03 --max-heldout-eval 600"
  for r in margin nearest; do
    CMD="$CMD && python tools/rl_prep/build_margin_preferences.py --root $E/world_action_full_v1 --geom-root $E/world_action_geom_full_v1 --manifest $RUN/manifest_v2.json --norm-stats $NORM --winner-rule $r --output-dir $RUN/prefs_$r"
  done
  J=$(sbatch --parsable --job-name=scaleup_prep --account=xinxia0 --partition=standard --cpus-per-task=4 --mem=32g --time=02:00:00 \
      --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err --wrap "$CMD")
  echo "prep job $J   run dir $RUN   (then: bash scripts/hpc/run_scaleup_navtest.sh stage2 $RUN)"
  exit 0
fi
RUN="${2:?RUN_DIR from stage1}"
for f in "$RUN/manifest_v2.json" "$RUN/prefs_margin/pairs.jsonl" "$RUN/prefs_nearest/pairs.jsonl"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
export SCALED_RUN_DIR="$RUN" PAIRS_MARGIN="$RUN/prefs_margin/pairs.jsonl" PAIRS_NEAREST="$RUN/prefs_nearest/pairs.jsonl" PAIRS_PDMS="$RUN/prefs_margin/pairs.jsonl"
JOBS=()
for a in marginpref nearestpref; do
  J=$(sbatch --parsable --array=0-0 --time=08:00:00 --export=ALL,SCALED_MODE_NAME=$a,SCALED_SEED=20,SCALED_NAME=scaleup_${a}_s20,SCALED_UPDATES=1600,SCALED_EVAL_SCENES=100 scripts/hpc/submit_pair_pref_train.slurm)
  echo "$a -> $J"; JOBS+=("$J")
done
DEP="afterok:$(IFS=:; echo "${JOBS[*]}")"
OUT=$X/official_navtest_trained_eval
LIST=$OUT/arms_scaleup.txt
[ -e "$LIST" ] && { echo "refusing existing $LIST"; exit 1; }
for a in marginpref nearestpref; do echo "scaleup_${a}_s20|$RUN/scaleup_${a}_s20/action_policy.pt" >> "$LIST"; done
export EVAL_LIST="$LIST" EVAL_OUT="$OUT" EVAL_SHARDS=12 NT_COHORT=$X/official_navtest_full_cohort2
J=$(sbatch --parsable --array=0-23%5 --dependency="$DEP" scripts/hpc/submit_navtest_policy_eval.slurm)
echo "eval array $J (after $DEP); output $OUT"
