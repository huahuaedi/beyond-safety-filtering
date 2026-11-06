#!/bin/bash
# Infer bounded pseudo-controls and audit trajectory reconstruction.

#SBATCH --job-name=pseudo_controls
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12g
#SBATCH --time=08:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"

INPUT_ROOT="${CONTROL_INPUT_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_full_v1}"
DYNAMICS_ROOT="${CONTROL_DYNAMICS_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_dynamics_full_v2}"
OUTPUT_ROOT="${CONTROL_OUTPUT_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_controls_full_v1}"
REFINE_STEPS="${CONTROL_REFINE_STEPS:-20}"
MAX_SCENES="${CONTROL_MAX_SCENES:-0}"

echo "pseudo-control shard=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT refine=$REFINE_STEPS max_scenes=$MAX_SCENES"
srun python tools/rl_prep/build_pseudo_control_sidecars.py \
  "$INPUT_ROOT" "$DYNAMICS_ROOT" "$OUTPUT_ROOT" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --refine-steps "$REFINE_STEPS" \
  --max-scenes "$MAX_SCENES"
