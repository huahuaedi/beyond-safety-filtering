#!/bin/bash
# Usage: bash scripts/hpc/submit_pair_pref.sh smoke|arms|all <RUN_DIR>
#   all: smoke, then the six arms behind it (afterok), then the summary; nothing starts if the smoke fails.
# Needs: pairs.jsonl for both winner rules (built by tools/rl_prep/build_margin_preferences.py) and the scaled run dir.
set -euo pipefail
STAGE="${1:?smoke|arms|all}"
RUN_DIR="${2:?RUN_DIR (the offline_scaled_* directory with manifest_v2.json)}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
cd "$ROOT"
export SCALED_RUN_DIR="$RUN_DIR"
export PAIRS_MARGIN="${PAIRS_MARGIN:-$ROOT/output/prefs_v3_margin/pairs.jsonl}"
export PAIRS_PDMS="${PAIRS_PDMS:-$ROOT/output/prefs_v3_pdms/pairs.jsonl}"
export PAIRS_NEAREST="${PAIRS_NEAREST:-$ROOT/output/prefs_v3_nearest/pairs.jsonl}"
for f in "$RUN_DIR/manifest_v2.json" "$PAIRS_MARGIN" "$PAIRS_PDMS" "$PAIRS_NEAREST"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
if [ "$STAGE" = all ]; then
  [ -e "$RUN_DIR/smoke_pair_pref" ] && { echo "refusing existing $RUN_DIR/smoke_pair_pref"; exit 1; }
  for a in marginpref pdmspref nearestpref; do for s in 20 21; do
    [ -e "$RUN_DIR/scaled_v1_${a}_seed${s}" ] && { echo "refusing existing $RUN_DIR/scaled_v1_${a}_seed${s}"; exit 1; }
  done; done
  JS=$(sbatch --parsable --array=0-0 --time=01:30:00 \
      --export=ALL,SCALED_MODE_NAME=marginpref,SCALED_SEED=20,SCALED_NAME=smoke_pair_pref,SCALED_UPDATES=2,SCALED_EVAL_SCENES=2 \
      scripts/hpc/submit_pair_pref_train.slurm)
  echo "smoke job $JS"
  J=$(sbatch --parsable --dependency=afterok:$JS --array=0-5%4 scripts/hpc/submit_pair_pref_train.slurm)
  echo "arms array $J (starts only if the smoke succeeds)"
  S=$(sbatch --parsable --dependency=afterany:$J --job-name=pair_summary --account=xinxia0 --partition=standard --time=00:20:00 \
      --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
      --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $ROOT/tools/rl_prep && python summarize_offline_scaled.py $RUN_DIR --out-name pair_pref_comparison.json")
  echo "summary job $S   result $RUN_DIR/pair_pref_comparison.json"
  exit 0
fi
if [ "$STAGE" = smoke ]; then
  [ -e "$RUN_DIR/smoke_pair_pref" ] && { echo "refusing existing $RUN_DIR/smoke_pair_pref"; exit 1; }
  J=$(sbatch --parsable --array=0-0 --time=01:30:00 \
      --export=ALL,SCALED_MODE_NAME=marginpref,SCALED_SEED=20,SCALED_NAME=smoke_pair_pref,SCALED_UPDATES=2,SCALED_EVAL_SCENES=2 \
      scripts/hpc/submit_pair_pref_train.slurm)
  echo "smoke job $J   afterwards check $RUN_DIR/smoke_pair_pref/summary.json"
  exit 0
fi
for a in marginpref pdmspref nearestpref; do for s in 20 21; do
  [ -e "$RUN_DIR/scaled_v1_${a}_seed${s}" ] && { echo "refusing existing $RUN_DIR/scaled_v1_${a}_seed${s}"; exit 1; }
done; done
[ -d "$RUN_DIR/smoke_pair_pref/validation" ] || { echo "run the smoke stage first"; exit 1; }
J=$(sbatch --parsable --array=0-5%4 scripts/hpc/submit_pair_pref_train.slurm)
echo "arms array $J"
S=$(sbatch --parsable --dependency=afterany:$J --job-name=pair_summary --account=xinxia0 --partition=standard --time=00:20:00 \
    --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
    --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $ROOT/tools/rl_prep && python summarize_offline_scaled.py $RUN_DIR --out-name pair_pref_comparison.json")
echo "summary job $S   result $RUN_DIR/pair_pref_comparison.json"
