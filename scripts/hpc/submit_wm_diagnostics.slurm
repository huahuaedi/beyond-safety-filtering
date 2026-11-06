#!/bin/bash
#SBATCH --job-name=wm_diag
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --array=0-14%5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err
set -euo pipefail
cd "$DRIVEVLA_REPO_ROOT"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
python -c 'import sys, numpy; print("runtime:", sys.executable, "numpy", numpy.__version__, flush=True)'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WM_RISK_EPOCHS=40
srun python tools/rl_prep/run_wm_diagnostics.py --fold "$((SLURM_ARRAY_TASK_ID % 5))" --seed "$((SLURM_ARRAY_TASK_ID / 5))" --output "$WM_DIAGNOSTIC_OUTPUT"
