#!/bin/bash
# Paper ablations for the waypoint-control RL mainline.
# Tasks 0-2: no risk replay, adapter enabled.
# Tasks 3-5: risk replay enabled, no control adapter.
# Submit with: sbatch --array=0-5%3 scripts/hpc/submit_waypoint_control_rl_ablations.slurm

#SBATCH --job-name=wpctrl_ablate
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/%x-%A_%a.log
#SBATCH --error=/home/%u/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="$HOME/DriveVLA-W0"
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as an array}"
SEED=$((TASK_ID % 3))
VARIANT_ID=$((TASK_ID / 3))

export RL_SEED="$SEED"
export RL_NUM_STEPS=25
export RL_SAVE_STEPS=25
export RL_ROLLOUT_MICRO_BATCH=1

case "$VARIANT_ID" in
    0)
        VARIANT="no_replay"
        export RL_RISK_REPLAY_EVERY=0
        export RL_CONTROL_ADAPTER_CHECKPOINT="$REPO_ROOT/output/waypoint_control_adapter_v1/seed_0/model.pt"
        export RL_CONTROL_ADAPTER_MODE=infeasible_only
        ;;
    1)
        VARIANT="no_adapter"
        export RL_RISK_REPLAY_EVERY=1
        export RL_CONTROL_ADAPTER_CHECKPOINT=""
        ;;
    *)
        echo "invalid array task $TASK_ID" >&2
        exit 2
        ;;
esac

export RL_CHECKPOINT_DIR="$REPO_ROOT/pretrained_models/DriveVLA-W0/waypoint_control_${VARIANT}_seed${SEED}"

echo "ABLATION variant=$VARIANT seed=$SEED steps=$RL_NUM_STEPS replay_every=$RL_RISK_REPLAY_EVERY adapter='${RL_CONTROL_ADAPTER_CHECKPOINT}' checkpoint_dir=$RL_CHECKPOINT_DIR"

# Resource allocation comes from this wrapper; the sourced file supplies the
# validated environment and training command.
source "$REPO_ROOT/scripts/hpc/submit_rl_stage1.slurm"
