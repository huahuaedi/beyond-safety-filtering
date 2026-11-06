#!/bin/bash
# Build kinematic/dynamics-feasibility sidecars for full world-action data.

#SBATCH --job-name=dynamics_audit
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"

INPUT_ROOT="${DYNAMICS_INPUT_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_full_v1}"
OUTPUT_ROOT="${DYNAMICS_OUTPUT_ROOT:-$REPO_ROOT/navsim_data/exp/world_action_dynamics_full_v2}"
DT="${DYNAMICS_DT:-0.5}"

echo "dynamics shard=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT input=$INPUT_ROOT output=$OUTPUT_ROOT dt=$DT"
srun python tools/rl_prep/audit_world_action_dynamics.py \
  "$INPUT_ROOT" "$OUTPUT_ROOT" \
  --dt "$DT" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT"
