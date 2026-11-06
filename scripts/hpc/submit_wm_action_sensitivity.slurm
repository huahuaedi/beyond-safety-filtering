#!/bin/bash
# Does the frozen WM latent respond to the candidate action?  Reads the
# already-extracted latents and labels; needs no GPU.  Set WM_SENS_OUTPUT to a
# new json path (the script refuses to overwrite).

#SBATCH --job-name=wm_action_sens
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

EXP="$REPO_ROOT/navsim_data/exp"
: "${WM_SENS_OUTPUT:?set WM_SENS_OUTPUT to a new results json path}"
python tools/rl_prep/wm_action_sensitivity.py \
    --data-root "${WM_SENS_DATA_ROOT:-$EXP/world_action_full_v1}" \
    --latent-root "${WM_SENS_LATENT_ROOT:-$EXP/world_action_latents_full_v1}" \
    --max-scenes "${WM_SENS_MAX_SCENES:-300}" \
    --output "$WM_SENS_OUTPUT"
