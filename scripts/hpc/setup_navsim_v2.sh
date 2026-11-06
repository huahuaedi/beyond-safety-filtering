#!/bin/bash
# Separate NAVSIM v2 environment (EPDMS scorer) next to the existing v1 'navsim' environment, which is left untouched.
# Run as a batch job (the login node kills conda while it loads the package index):   sbatch scripts/hpc/setup_navsim_v2.slurm
set -euo pipefail
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$HOME"
[ -d navsim_v2 ] || git clone --depth 1 https://github.com/autonomousvision/navsim.git navsim_v2
cd navsim_v2
git log --oneline -1
# libmamba needs far less memory than the classic solver; fall back to the classic one if it is not installed
conda env list | grep -q "^navsim2 " || conda env create --solver=libmamba -n navsim2 -f environment.yml || conda env create -n navsim2 -f environment.yml
conda activate navsim2
pip install -e .
python - <<'PY'
import nuplan, navsim
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
print("nuplan", nuplan.__file__)
print("navsim", navsim.__file__)
print("v2 scorer import OK")
PY
echo SETUP_DONE
