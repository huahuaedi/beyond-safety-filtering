#!/bin/bash
# Read-only inventory. Does not install, download, or claim benchmark readiness.
set -euo pipefail
repo="${NAVSIM_V2_REPO:-$HOME/navsim_v2}"
py="${NAVSIM_V2_PYTHON:-$HOME/.conda/envs/navsim2/bin/python}"
[[ -d "$repo/.git" && -x "$py" ]] || { echo 'Missing NAVSIM_V2_REPO or NAVSIM_V2_PYTHON'; exit 2; }
git -C "$repo" rev-parse HEAD
"$py" -c 'import sys, navsim; print(sys.executable); print(navsim.__file__)'
for key in NAVSIM_V2_DATA_ROOT NAVSIM_V2_CACHE_ROOT; do
 value="${!key:-}"
 if [[ -z "$value" || ! -d "$value" ]]; then
  echo "$key missing: provide the actual two-stage data/cache directory"; exit 2
 fi
 echo "$key=$value"
 ls -ld "$value"
done
find "$repo/navsim" -type f \( -iname '*two_stage*' -o -iname '*epdm*' -o -iname '*navhard*' \) -print
echo 'Inventory only: verify split, both stages, image/metadata coverage, cache schema and model input adapter before submitting evaluation.'
