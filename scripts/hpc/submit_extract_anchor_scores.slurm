#!/bin/bash
#SBATCH --job-name=anchor_scores
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128g
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
# Read-only: per-anchor sub-scores of the 7,171-scene candidate pool. Env: ANCHOR_OUT (new output dir, must not exist).
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
DATA_REPO="${DRIVEVLA_REPO_ROOT:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
OUT="${ANCHOR_OUT:?set ANCHOR_OUT}"
SCORES="${ANCHOR_SCORES:-$DATA_REPO/data/navsim/processed_data/anchor/formatted_pdm_score_8192.npy}"
POOL="$ROOT/navsim_data/exp/world_action_full_v1"
test -f "$SCORES" && test -d "$POOL"
test ! -e "$OUT" && mkdir -p "$OUT"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
python -u tools/rl_prep/extract_anchor_score_pool.py --scores "$SCORES" --pool-root "$POOL" \
  --output "$OUT/anchor_scores_pool.npz" --report "$OUT/anchor_scores_pool_report.json"
