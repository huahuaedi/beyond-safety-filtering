#!/bin/bash
# Stage 1: where do NAVSIM failures happen -- in the planned trajectory or in
# its execution by the tracker + vehicle model?
#
#   bash scripts/hpc/submit_execution_trace_pipeline.sh smoke   # 3 scenes, traced scores verified against pdm_score()
#   bash scripts/hpc/submit_execution_trace_pipeline.sh full    # 4 shards x TRACE_SCENES_PER_SHARD scenes + analysis
#
# Run from an authenticated Great Lakes shell after synchronising the repo.
# Decision rules are fixed in docs/archive/execution_trace_stage1.md before any result.
set -euo pipefail
STAGE="${1:-}"
case "$STAGE" in smoke|full) ;; *) echo "usage: $0 smoke|full" >&2; exit 2 ;; esac
export DRIVEVLA_REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$DRIVEVLA_REPO_ROOT"
command -v sbatch >/dev/null
RUN_ID="${TRACE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export TRACE_RUN_ROOT="${TRACE_RUN_ROOT:-$DRIVEVLA_REPO_ROOT/navsim_data/exp/execution_trace_${STAGE}_$RUN_ID}"
export TRACE_SUMMARY_DIR="${TRACE_SUMMARY_DIR:-$DRIVEVLA_REPO_ROOT/results_summary/execution_trace_${STAGE}_$RUN_ID}"
for path in "$TRACE_RUN_ROOT" "$TRACE_SUMMARY_DIR"; do
    [ ! -e "$path" ] || { echo "refusing to mix results in existing path: $path" >&2; exit 1; }
done

# Synthetic unit tests first (no model, NAVSIM or GPU needed).
PYTHONPATH="$DRIVEVLA_REPO_ROOT/tools/rl_prep${PYTHONPATH:+:$PYTHONPATH}" \
    python -m unittest test_execution_trace test_posttrain_pipeline

mkdir -p "$TRACE_RUN_ROOT" "$TRACE_SUMMARY_DIR"
git rev-parse HEAD > "$TRACE_SUMMARY_DIR/source_commit.txt"
git diff --binary HEAD > "$TRACE_SUMMARY_DIR/source_changes.patch"

if [ "$STAGE" = smoke ]; then
    collect=$(TRACE_VERIFY=1 TRACE_SCENES_PER_SHARD=3 sbatch --parsable --array=0-0 --time=01:00:00 \
        scripts/hpc/submit_execution_trace_collect.slurm)
else
    collect=$(TRACE_VERIFY=0 TRACE_SCENES_PER_SHARD="${TRACE_SCENES_PER_SHARD:-500}" sbatch --parsable \
        scripts/hpc/submit_execution_trace_collect.slurm)
fi
collect=${collect%%;*}
analyze=$(sbatch --parsable --dependency="afterok:$collect" scripts/hpc/submit_execution_trace_analyze.slurm)
analyze=${analyze%%;*}
printf 'stage=%s collect=%s analyze=%s run_root=%s\n' "$STAGE" "$collect" "$analyze" "$TRACE_RUN_ROOT" \
    | tee "$TRACE_SUMMARY_DIR/jobs.txt"
if [ "$STAGE" = smoke ]; then
    echo "Smoke passes when: the collect log has no 'traced scores differ' error, 3 files exist under"
    echo "  $TRACE_RUN_ROOT/traces, and $TRACE_SUMMARY_DIR/report.json shows"
    echo "  sanity.scorer_flag_vs_dac_agreement == 1.0, n_row_errors == 0, n_open_loop_errors == 0."
fi
echo "results: $TRACE_SUMMARY_DIR/report.json (when the analysis job finishes)"
