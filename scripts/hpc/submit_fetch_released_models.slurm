#!/bin/bash
#SBATCH --job-name=fetch_models
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
# Released DriveVLA-W0 checkpoints missing on this cluster (Query-based 88.4, Qwen 85.3) plus the authors' 87.2 json outputs.
# Env: FETCH_ARGS (extra arguments for fetch_released_models.py, e.g. "--dry-run").
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
DEST=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data/pretrained_models/DriveVLA-W0
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
python -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub
python "$ROOT/scripts/hpc/fetch_released_models.py" "$DEST" --json-zip ${FETCH_ARGS:-}
