#!/bin/bash
# Usage: bash scripts/hpc/submit_navtest_policy_eval.sh smoke|full [RUN_DIR]
#   smoke: 8 scenes of the base policy and of one checkpoint.   full: baseline + every saved preference arm, 1000 scenes each.
# EVAL_DEP=afterok:<job>[:<job>] delays the array until those jobs (e.g. extra training seeds) have succeeded; the arm list is
# read when this script runs; arms that do not exist yet are added with EVAL_EXTRA="marginpref_seed22 ...".
set -euo pipefail
STAGE="${1:?smoke|full}"
RUN="${2:-$HOME/DriveVLA-Safety-RL-offline-20260907/output/offline_scaled_20261008_150131}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
C="${NT_COHORT:-$ASSET/pretrained_models/DriveVLA-W0/posttrain_runs/rl_pareto_navtest_v2_20261005_133703_37354}"
cd "$ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ASSET/navsim_data/exp/navtest_policy_eval_${STAMP}_${STAGE}"
mkdir "$OUT"
LIST="$OUT/arms.txt"
echo "baseline|-" > "$LIST"
for d in "$RUN"/scaled_v1_marginpref_seed* "$RUN"/scaled_v1_nearestpref_seed* "$RUN"/scaled_v1_pdmspref_seed*; do
  [ -f "$d/action_policy.pt" ] || continue
  echo "$(basename "$d" | sed 's/^scaled_v1_//')|$d/action_policy.pt" >> "$LIST"
done
# EVAL_EXTRA="marginpref_seed22 ..." lists arms whose checkpoints are still being trained; use together with EVAL_DEP.
for label in ${EVAL_EXTRA:-}; do
  grep -q "^$label|" "$LIST" || echo "$label|$RUN/scaled_v1_$label/action_policy.pt" >> "$LIST"
done
[ "$STAGE" = full ] && N=$(wc -l < "$LIST") || { head -n 2 "$LIST" > "$LIST.tmp"; mv "$LIST.tmp" "$LIST"; N=2; }
echo "arms ($N):"; cat "$LIST"
export EVAL_LIST="$LIST" EVAL_OUT="$OUT"
[ "$STAGE" = smoke ] && export EVAL_LIMIT=8
DEP=()
[ -z "${EVAL_DEP:-}" ] || DEP=(--dependency="$EVAL_DEP")
J=$(sbatch --parsable --array=0-$((N-1))%4 ${DEP[@]+"${DEP[@]}"} scripts/hpc/submit_navtest_policy_eval.slurm)
echo "eval array $J   output $OUT"
if [ "$STAGE" = full ]; then
  RUNS=$(awk -F'|' -v o="$OUT" '{printf "%s=%s/%s ", $1, o, $1}' "$LIST")
  S=$(sbatch --parsable --dependency=afterok:$J --job-name=nt_policy_summary --account=xinxia0 --partition=standard --cpus-per-task=2 --mem=8g --time=00:20:00 \
      --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
      --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $ROOT/tools/rl_prep && python summarize_policy_eval.py --runs $RUNS --token-log-map $C/token_log_map.json --output $OUT/summary.json")
  echo "summary job $S   result $OUT/summary.json"
fi
