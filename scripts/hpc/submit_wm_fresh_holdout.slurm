#!/bin/bash
#SBATCH --job-name=wm_fresh_holdout
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
set -euo pipefail
cd "$DRIVEVLA_REPO_ROOT"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -c 'import sys,numpy; print(sys.executable,numpy.__version__,flush=True)'
srun python tools/rl_prep/evaluate_wm_fresh_holdout.py --source "$HOME/DriveVLA-W0/navsim_data/exp" --checkpoints "$DRIVEVLA_REPO_ROOT/output/wm_diagnostics_20260923_154646_envfix" --output "$WM_FRESH_OUTPUT"
