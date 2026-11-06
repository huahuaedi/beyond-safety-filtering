#!/bin/bash
# TTC block + world-model PCA block -> selector comparison, all CPU.  Usage:
#   SELECTOR_MANIFEST=/path/manifest_v2.json bash scripts/hpc/submit_selector_upgrade.sh smoke|full
# smoke: 20 scenes per block into *_smoke dirs, analysis is NOT run (it needs the full blocks).
set -euo pipefail
MODE="${1:?smoke|full}"
: "${SELECTOR_MANIFEST:?set SELECTOR_MANIFEST}"
[ -f "$SELECTOR_MANIFEST" ] || { echo "no manifest $SELECTOR_MANIFEST" >&2; exit 1; }
cd "$(dirname "$0")/../.."
export SELECTOR_MANIFEST
case "$MODE" in
  smoke)
    t=$(GEOM_TTC=1 GEOM_TARGETS=fresh GEOM_LIMIT=20 GEOM_OUT_SUFFIX=_smoke sbatch --parsable --array=0-0 scripts/hpc/submit_extract_geometric_features.slurm)
    w=$(WM_PCA_LIMIT=40 WM_PCA_SUFFIX=_smoke sbatch --parsable scripts/hpc/submit_wm_pca_features.slurm)
    echo "smoke jobs: ttc=$t wm_pca=$w" ;;
  full)
    t=$(GEOM_TTC=1 GEOM_TARGETS=fresh sbatch --parsable scripts/hpc/submit_extract_geometric_features.slurm)
    w=$(sbatch --parsable scripts/hpc/submit_wm_pca_features.slurm)
    a=$(sbatch --parsable --dependency=afterok:$t:$w scripts/hpc/submit_selector_extras.slurm)
    echo "full jobs: ttc=$t wm_pca=$w analysis=$a (runs after both succeed)" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
