#!/bin/bash
# Run from an authenticated Great Lakes shell after synchronizing the repair.
set -euo pipefail
export DRIVEVLA_REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$DRIVEVLA_REPO_ROOT"
export WM_ACTION_DATA_ROOT="${WM_ACTION_DATA_ROOT:-$DRIVEVLA_REPO_ROOT/navsim_data/exp/world_action_v1}"
export WM_LATENT_ROOT="${WM_LATENT_ROOT:-$DRIVEVLA_REPO_ROOT/navsim_data/exp/world_action_latents_v1}"
RUN_ID="${WM_REPAIR_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export WM_RISK_RANKER_OUTPUT="${WM_RISK_RANKER_OUTPUT:-$DRIVEVLA_REPO_ROOT/output/wm_review_repair_$RUN_ID}"
export WM_MANIFEST_OUTPUT="${WM_MANIFEST_OUTPUT:-$DRIVEVLA_REPO_ROOT/results_summary/wm_review_repair_$RUN_ID}"
# The array mapping in the CV script is five folds x four feature modes.
export WM_RISK_CV_FOLDS=5
command -v sbatch >/dev/null
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['WM_ACTION_DATA_ROOT'])
latent = Path(os.environ['WM_LATENT_ROOT'])
seen = set()
for split in ('train', 'val'):
    files = sorted((root / split).glob('*.npz'))
    if not files:
        raise SystemExit(f'No candidate files: {root / split}')
    for path in files:
        if path.stem in seen:
            raise SystemExit(f'Duplicate scene token: {path.stem}')
        seen.add(path.stem)
        if not (latent / split / path.name).is_file():
            raise SystemExit(f'Missing latent: {latent / split / path.name}')
    print(f'{split}: {len(files)} candidate/latent pairs')
for key in ('WM_RISK_RANKER_OUTPUT', 'WM_MANIFEST_OUTPUT'):
    path = Path(os.environ[key])
    if path.exists():
        raise SystemExit(f'Refusing to mix results in existing output directory: {path}')
PY
# Lightweight numerical tests only; all ranker training runs through Slurm.
PYTHONPATH="$DRIVEVLA_REPO_ROOT/tools/rl_prep${PYTHONPATH:+:$PYTHONPATH}" \
    python -m unittest test_review_regressions test_policy_optimization \
    test_adaptive_reward test_risk_curriculum test_rollout_diagnostics
mkdir -p "$WM_RISK_RANKER_OUTPUT" "$WM_MANIFEST_OUTPUT"
git rev-parse HEAD > "$WM_RISK_RANKER_OUTPUT/source_commit.txt"
git diff --binary HEAD > "$WM_RISK_RANKER_OUTPUT/source_changes.patch"
# Include untracked sources in provenance without touching Git's index.
python - <<'PY'
import hashlib, json, os
from pathlib import Path
paths = sorted(Path('tools/rl_prep').glob('*.py')) + sorted(Path('scripts/hpc').glob('*'))
checksums = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
(Path(os.environ['WM_RISK_RANKER_OUTPUT']) / 'source_sha256.json').write_text(json.dumps(checksums, indent=2))
PY
cv=$(sbatch --parsable scripts/hpc/submit_world_action_risk_ranker_cv.slurm)
cv=${cv%%;*}
printf 'CV array submitted: %s\n' "$cv"
printf '%s\n' "$cv" > "$WM_RISK_RANKER_OUTPUT/cv_job_id.txt"
if manifest=$(sbatch --parsable --dependency="afterok:$cv" scripts/hpc/submit_build_wm_risk_manifests.slurm); then
    manifest=${manifest%%;*}
    printf 'Mining/manifest job submitted: %s (afterok:%s)\n' "$manifest" "$cv"
    printf '%s\n' "$manifest" > "$WM_RISK_RANKER_OUTPUT/manifest_job_id.txt"
else
    printf 'Manifest submission failed; CV job %s remains submitted. Do not resubmit the whole pipeline.\n' "$cv" >&2
    exit 1
fi
printf 'Ranker results: %s\nManifests: %s\n' "$WM_RISK_RANKER_OUTPUT" "$WM_MANIFEST_OUTPUT"
squeue -u "$USER"
