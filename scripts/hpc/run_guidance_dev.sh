#!/bin/bash
# Development experiment for geometric cost guidance, on the 600 held-out-log pool scenes (never on navtest).
# Usage:  bash scripts/hpc/run_guidance_dev.sh smoke|dev
#   hard:  scenes in HARD_TOKENS (json) where selection cannot repair a failure; lambda in {0, 0.1, 0.2, 0.4}; candidates only
#          (analyse with tools/rl_prep/analyze_guided_repair.py).
#   smoke: 8 scenes; lambda in {0, 1e-9, 0.3}.  1e-9 must reproduce lambda = 0 (loop regression); 0.3 shows that guidance acts.
#   dev:   600 scenes; lambda in {0, 0.1, 0.2, 0.4}; guided candidates -> geometry features -> the selector fitted on the
#          manifest's training scenes only -> selector_dev.json per lambda.
set -euo pipefail
STAGE="${1:?smoke|dev|hard}"
OFF="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
RUN="$OFF/output/offline_scaled_20261008_150131"
EXP="$OFF/navsim_data/exp"
CACHE="$EXP/metric_cache_trainval_rl"
PKL="$ASSET/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
W="$ASSET/navsim_data/exp/guidance_dev_$(date +%Y%m%d_%H%M%S)_${STAGE}"
mkdir "$W"; cd "$OFF"
if [ "$STAGE" = hard ]; then cp "${HARD_TOKENS:?set HARD_TOKENS to the json from select_hard_scenes.py}" "$W/tokens.json"; else
python3 - "$RUN/manifest_v2.json" "$W/tokens.json" "$STAGE" <<'PY'
import json, sys
tokens = json.load(open(sys.argv[1]))["heldout_eval_tokens"]
json.dump(tokens[:8] if sys.argv[3] == "smoke" else tokens, open(sys.argv[2], "w"))
PY
fi
python3 tools/rl_prep/make_token_log_map.py --tokens-json "$W/tokens.json" --metric-cache "$CACHE" --output "$W/token_log_map.json"
JE=$(GUIDE_TOKENS="$W/tokens.json" GUIDE_CACHE="$CACHE" GUIDE_OUT="$W/guide_inputs" DRIVEVLA_REPO_ROOT="$OFF" \
     sbatch --parsable --array=0-3 scripts/hpc/submit_extract_guidance_inputs.slurm)
echo "guidance inputs job $JE   (workdir $W)"
if [ "$STAGE" = smoke ]; then LAMBDAS="0 1e-9 0.3"; ARR="0-0"; elif [ "$STAGE" = hard ]; then LAMBDAS="0 0.1 0.2 0.4"; ARR="0-1"; else LAMBDAS="0 0.1 0.2 0.4"; ARR="0-3"; fi
for L in $LAMBDAS; do
  OUT="$W/lam_$L"; mkdir -p "$OUT"
  JC=$(GD_OUT="$OUT" GD_TOKENS="$W/tokens.json" GD_SCENE_PKL="$PKL" GD_CACHE="$CACHE" GD_GUIDE_DIR="$W/guide_inputs" GD_LAMBDA="$L" RL_SEED=0 \
       sbatch --parsable --array=$ARR --dependency=afterok:$JE scripts/hpc/submit_guided_candidates.slurm)
  echo "lambda $L: collection job $JC   out $OUT"
  [ "$STAGE" != dev ] && continue
  JG=$(DRIVEVLA_REPO_ROOT="$OFF" GEOM_TARGETS=fresh WM_FRESH_DATA_ROOT="$OUT/collection" WM_FRESH_GEOM_ROOT="$OUT/geom" NAVSIM_METRIC_CACHE_PATH="$CACHE" \
       sbatch --parsable --dependency=afterok:$JC scripts/hpc/submit_extract_geometric_features.slurm)
  JF=$(sbatch --parsable --dependency=afterok:$JG --job-name=guide_select --account=xinxia0 --partition=standard --cpus-per-task=8 --mem=64g --time=02:00:00 \
       --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
       --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $OFF/tools/rl_prep && python fit_and_apply_selector.py --train-root $EXP/world_action_full_v1 --train-geom-root $EXP/world_action_geom_full_v1 --train-manifest $RUN/manifest_v2.json --apply-root $OUT/collection --apply-geom-root $OUT/geom --token-log-map $W/token_log_map.json --output $OUT/selector_dev.json")
  echo "lambda $L: geometry $JG, selector $JF -> $OUT/selector_dev.json"
done
echo "workdir: $W"
