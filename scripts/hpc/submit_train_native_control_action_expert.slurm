#!/bin/bash
# Warm-start DriveVLA's Action Expert in native (acceleration, curvature-rate) space.

#SBATCH --job-name=native_ctrl_ae
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --time=12:00:00
# %A is the array's master job ID and %a is its task index. For a non-array
# submission, Slurm substitutes a sentinel value for %a, which is still unique.
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
# Compute-node module initialization can place /sw's Python ahead of the
# activated environment.  Force the environment interpreter, matching the
# known-good RL submission script.
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"

# An explicit seed still wins; otherwise each array task gets its own seed.
export NATIVE_CONTROL_SEED="${NATIVE_CONTROL_SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
export NATIVE_CONTROL_EPOCHS="${NATIVE_CONTROL_EPOCHS:-1}"
export NATIVE_CONTROL_MAX_SCENES="${NATIVE_CONTROL_MAX_SCENES:-16}"
export NATIVE_CONTROL_MAX_UPDATES="${NATIVE_CONTROL_MAX_UPDATES:-2}"
export NATIVE_CONTROL_LR="${NATIVE_CONTROL_LR:-1e-5}"
export NATIVE_CONTROL_CANDIDATE_SELECTION="${NATIVE_CONTROL_CANDIDATE_SELECTION:-quality_top1}"
export NATIVE_CONTROL_DATA_ROOT="${NATIVE_CONTROL_DATA_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_full_v1}"
export NATIVE_CONTROL_SIDECAR_ROOT="${NATIVE_CONTROL_SIDECAR_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_controls_full_v1}"
export NATIVE_CONTROL_OUTPUT="${NATIVE_CONTROL_OUTPUT:-$REPO_ROOT/output/native_control_action_expert_v1_smoke}"

echo "seed=$NATIVE_CONTROL_SEED epochs=$NATIVE_CONTROL_EPOCHS max_scenes=$NATIVE_CONTROL_MAX_SCENES max_updates=$NATIVE_CONTROL_MAX_UPDATES candidate_selection=$NATIVE_CONTROL_CANDIDATE_SELECTION"
echo "python=$(command -v python)"
python -c 'import transformers; print("transformers=" + transformers.__version__); import transformers.cache_utils'
srun "$CONDA_PREFIX/bin/python" tools/rl_prep/train_native_control_action_expert.py
