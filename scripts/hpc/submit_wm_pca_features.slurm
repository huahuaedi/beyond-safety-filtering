#!/bin/bash
# Low-dimensional world-model features (PCA of scene-centred end-of-action latents), fitted on
# the log-disjoint training scenes only.  CPU only, memory heavy (reads ~15 GB of latents per
# layer).  Writes <WM_PCA_ROOT>/L<i>/<split>/<token>.npz.
#
# Smoke: WM_PCA_LIMIT=40 WM_PCA_SUFFIX=_smoke sbatch scripts/hpc/submit_wm_pca_features.slurm

#SBATCH --job-name=wm_pca_feat
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160g
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"

EXP="$REPO_ROOT/navsim_data/exp"
LATENTS="${WM_LATENT_ROOT:-$EXP/world_action_latents_full_v1}"
OUT="${WM_PCA_ROOT:-$EXP/world_action_wm_pca_full_v1}${WM_PCA_SUFFIX:-}"
MANIFEST="${SELECTOR_MANIFEST:?SELECTOR_MANIFEST (manifest_v2.json of the log-disjoint split)}"
EXTRA=()
if [ "${WM_PCA_LIMIT:-0}" -gt 0 ]; then EXTRA+=(--limit "$WM_PCA_LIMIT"); fi

export OMP_NUM_THREADS=8
python tools/rl_prep/extract_wm_pca_features.py \
    --latent-root "$LATENTS" --manifest "$MANIFEST" --out-root "$OUT" \
    --layers "${WM_PCA_LAYERS:-0,1,2,3}" --k "${WM_PCA_K:-32}" ${EXTRA[@]+"${EXTRA[@]}"}
