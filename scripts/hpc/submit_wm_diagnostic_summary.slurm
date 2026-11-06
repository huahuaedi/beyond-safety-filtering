#!/bin/bash
#SBATCH --job-name=wm_diag_summary
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=4g
#SBATCH --time=00:10:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
set -euo pipefail
cd "$DRIVEVLA_REPO_ROOT"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
python -c 'import sys, numpy; print("runtime:", sys.executable, "numpy", numpy.__version__, flush=True)'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/rl_prep/summarize_wm_diagnostics.py "$WM_DIAGNOSTIC_OUTPUT"
