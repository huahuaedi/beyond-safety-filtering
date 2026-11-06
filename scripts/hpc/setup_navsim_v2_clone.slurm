#!/bin/bash
# Fallback for setup_navsim_v2.slurm: no solver. Clone the working v1 'navsim' environment and install the v2 sources into the clone.
# The v1 environment is not modified. Missing packages, if any, are reported by the import check at the end.
#SBATCH --job-name=navsim2_clone
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
set -euo pipefail
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
[ -d "$HOME/navsim_v2" ] || git clone --depth 1 https://github.com/autonomousvision/navsim.git "$HOME/navsim_v2"
if conda env list | grep -q "^navsim2 "; then
  # an earlier (cancelled) job may have left a half-built environment; never reuse it silently
  NAV2_PY="$(conda env list | awk '$1=="navsim2"{print $NF}')/bin/python"
  "$NAV2_PY" -c "import nuplan" 2>/dev/null || { echo "navsim2 exists but cannot import nuplan: run 'conda env remove -n navsim2 -y' and resubmit"; exit 3; }
else
  conda create -y -n navsim2 --clone navsim
fi
conda activate navsim2
cd "$HOME/navsim_v2"
pip install -e . --no-deps
python - <<'PY'
import nuplan, navsim
print("nuplan", nuplan.__file__)
print("navsim", navsim.__file__)
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
print("v2 scorer import OK")
PY
echo CLONE_SETUP_DONE
