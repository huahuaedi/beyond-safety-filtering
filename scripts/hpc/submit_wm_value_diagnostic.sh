#!/bin/bash
# Does the world model add risk information beyond action and simple geometry?
#
#   bash scripts/hpc/submit_wm_value_diagnostic.sh smoke   # 20-scene geometry extraction on the real cache
#   bash scripts/hpc/submit_wm_value_diagnostic.sh full    # everything, dependency-chained
#
# Run from an authenticated Great Lakes shell after synchronising the repo.
# Read docs/archive/wm_value_diagnostic.md first: the decision rules are fixed there
# before any result exists.
set -euo pipefail
STAGE="${1:-}"
case "$STAGE" in smoke|full) ;; *) echo "usage: $0 smoke|full" >&2; exit 2 ;; esac
export DRIVEVLA_REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$DRIVEVLA_REPO_ROOT"
command -v sbatch >/dev/null
EXP="$DRIVEVLA_REPO_ROOT/navsim_data/exp"
RUN_ID="${WM_VALUE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [ "$STAGE" = smoke ]; then
    smoke_out="${WM_GEOM_ROOT:-$EXP/world_action_geom_v1}_smoke"
    [ ! -e "$smoke_out" ] || { echo "refusing to reuse $smoke_out" >&2; exit 1; }
    job=$(GEOM_LIMIT=20 GEOM_OUT_SUFFIX=_smoke GEOM_TARGETS=dev sbatch --parsable --array=0-0 \
        scripts/hpc/submit_extract_geometric_features.slurm)
    echo "smoke extraction job: ${job%%;*}"
    echo "When it finishes: the log's JSON summary must show missing_metric_cache=0 and"
    echo "moving_agent_fraction>0. Then check the features against the labels:"
    echo "  python tools/rl_prep/summarize_risk_baselines.py --audit-only \\"
    echo "      --data-root ${WM_ACTION_DATA_ROOT:-$EXP/world_action_v1} --geom-root $smoke_out"
    echo "Sane: min_clearance / offroad_frac_max clearly away from 0.5 for ttc/nc and dac respectively."
    exit 0
fi

export WM_ACTION_DATA_ROOT="${WM_ACTION_DATA_ROOT:-$EXP/world_action_v1}"
export WM_LATENT_ROOT="${WM_LATENT_ROOT:-$EXP/world_action_latents_v1}"
export WM_GEOM_ROOT="${WM_GEOM_ROOT:-$EXP/world_action_geom_v1}"
export WM_FRESH_DATA_ROOT="${WM_FRESH_DATA_ROOT:-$EXP/world_action_full_v1}"
export WM_FRESH_LATENT_ROOT="${WM_FRESH_LATENT_ROOT:-$EXP/world_action_latents_full_v1}"
export WM_FRESH_GEOM_ROOT="${WM_FRESH_GEOM_ROOT:-$EXP/world_action_geom_full_v1}"
export WM_RISK_RANKER_OUTPUT="${WM_RISK_RANKER_OUTPUT:-$DRIVEVLA_REPO_ROOT/output/wm_value_$RUN_ID}"
export WM_SUMMARY_OUTPUT="${WM_SUMMARY_OUTPUT:-$DRIVEVLA_REPO_ROOT/results_summary/wm_value_$RUN_ID/summary.json}"
export WM_SENS_OUTPUT="${WM_SENS_OUTPUT:-$DRIVEVLA_REPO_ROOT/results_summary/wm_value_$RUN_ID/action_sensitivity.json}"

for path in "$WM_RISK_RANKER_OUTPUT" "$WM_SUMMARY_OUTPUT" "$WM_SENS_OUTPUT"; do
    [ ! -e "$path" ] || { echo "refusing to mix results in existing path: $path" >&2; exit 1; }
done
for path in "$WM_ACTION_DATA_ROOT" "$WM_LATENT_ROOT" "$WM_FRESH_DATA_ROOT" "$WM_FRESH_LATENT_ROOT"; do
    [ -d "$path" ] || { echo "missing input directory: $path" >&2; exit 1; }
done

# Numerical/unit tests first (synthetic data; no model, NAVSIM or GPU needed).
PYTHONPATH="$DRIVEVLA_REPO_ROOT/tools/rl_prep${PYTHONPATH:+:$PYTHONPATH}" \
    python -m unittest test_geometric_features test_extract_geometric_features \
    test_wm_action_sensitivity test_risk_baselines test_ranker_pipeline

mkdir -p "$WM_RISK_RANKER_OUTPUT" "$(dirname "$WM_SUMMARY_OUTPUT")"
git rev-parse HEAD > "$WM_RISK_RANKER_OUTPUT/source_commit.txt"
git diff --binary HEAD > "$WM_RISK_RANKER_OUTPUT/source_changes.patch"

geom=$(sbatch --parsable scripts/hpc/submit_extract_geometric_features.slurm); geom=${geom%%;*}
sens=$(sbatch --parsable scripts/hpc/submit_wm_action_sensitivity.slurm); sens=${sens%%;*}
cv=$(sbatch --parsable --dependency="afterok:$geom" scripts/hpc/submit_risk_baselines_cv.slurm); cv=${cv%%;*}
sum=$(sbatch --parsable --dependency="afterok:$cv" scripts/hpc/submit_risk_baselines_summary.slurm); sum=${sum%%;*}
printf 'geometry=%s sensitivity=%s baselines_cv=%s summary=%s\n' "$geom" "$sens" "$cv" "$sum" \
    | tee "$WM_RISK_RANKER_OUTPUT/job_ids.txt"
echo "results: $(dirname "$WM_SUMMARY_OUTPUT")  (summary.md when the last job finishes)"
