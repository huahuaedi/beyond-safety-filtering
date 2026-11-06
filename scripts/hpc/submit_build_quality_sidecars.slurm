#!/bin/bash
# Build compact quality/advantage metadata after world-action collection.

#SBATCH --job-name=quality_sidecar
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail

REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"

INPUT_ROOT="${QUALITY_INPUT_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_full_v1}"
OUTPUT_ROOT="${QUALITY_OUTPUT_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_quality_full_v1}"
WEIGHT_MANIFEST="${QUALITY_WEIGHT_MANIFEST:-}"

ARGS=("$INPUT_ROOT" "$OUTPUT_ROOT")
if [[ -n "$WEIGHT_MANIFEST" ]]; then
  ARGS+=(--weight-manifest "$WEIGHT_MANIFEST")
fi

echo "input=$INPUT_ROOT output=$OUTPUT_ROOT weight_manifest=${WEIGHT_MANIFEST:-static}"
srun python tools/rl_prep/build_quality_conditioning_sidecars.py "${ARGS[@]}"
