#!/bin/bash
#SBATCH --job-name=stageA
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err
# Stage A candidate evaluators (docs/stageA_protocol.md). Env: STAGEA_MODE=tune|final, STAGEA_OUT (root), STAGEA_RUN_DIR (dir with manifest_v2.json),
# STAGEA_POOL_LOG_MAP, STAGEA_NAVTEST (selector output dir with collection/ geom/ stageA/navtest_tokens), STAGEA_COHORT (official cohort dir),
# for final also STAGEA_CONFIG (json from stageA_pick_config.py).
#   tune : array 0-19 = 5 variants x (lr 3e-4, 1e-3) x (dropout 0, 0.1), seed 0, scores `tune` only.
#   final: array 0-14 = 5 variants x seeds 0,1,2 with the chosen configuration, scores `decision` and `navtest`.
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
MODE="${STAGEA_MODE:?STAGEA_MODE=tune|final}"
OUT="${STAGEA_OUT:?STAGEA_OUT}"
RUN="${STAGEA_RUN_DIR:?STAGEA_RUN_DIR}"; LOGMAP="${STAGEA_POOL_LOG_MAP:?STAGEA_POOL_LOG_MAP}"
NT="${STAGEA_NAVTEST:?STAGEA_NAVTEST}"; COH="${STAGEA_COHORT:?STAGEA_COHORT}"
EXP="$ROOT/navsim_data/exp"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$ROOT/tools/rl_prep"
T=${SLURM_ARRAY_TASK_ID:-0}
VARS=(a b c d e)
COMMON=(--pool-root "$EXP/world_action_full_v1" --pool-geom "$EXP/world_action_geom_full_v1" --pool-tokens "$NT/stageA/pool_tokens"
        --manifest "$RUN/manifest_v2.json" --pool-log-map "$LOGMAP")
EXTRA=()
if [ "$MODE" = tune ]; then
  V=${VARS[$((T / 4))]}; C=$((T % 4)); LR=(3e-4 3e-4 1e-3 1e-3); DR=(0 0.1 0 0.1)
  NAME="${V}_lr${LR[$C]}_dr${DR[$C]}"
  [ "$V" = b ] && EXTRA+=(--match-to-c)
  python -u stageA_run.py --variant "$V" --seed 0 --lr "${LR[$C]}" --dropout "${DR[$C]}" --eval tune --out "$OUT/tune/$NAME" "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
else
  V=${VARS[$((T / 3))]}; S=$((T % 3))
  read LR DR < <(python - "$STAGEA_CONFIG" "$V" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))[sys.argv[2]]
print(c["lr"], c["dropout"])
PY
)
  [ "$V" = b ] && EXTRA+=(--match-to-c)
  [ "$V" = c ] && EXTRA+=(--shuffle-check)
  python -u stageA_run.py --variant "$V" --seed "$S" --lr "$LR" --dropout "$DR" --eval decision navtest --out "$OUT/final/${V}_seed${S}" "${COMMON[@]}" \
      --apply-root "$NT/collection" --apply-geom "$NT/geom" --apply-tokens "$NT/stageA/navtest_tokens" --apply-log-map "$COH/token_log_map.json" ${EXTRA[@]+"${EXTRA[@]}"}
fi
