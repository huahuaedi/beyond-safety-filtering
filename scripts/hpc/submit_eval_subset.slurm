#!/bin/bash
# Runs the same two-stage PDMS eval pipeline used to originally report the
# 87.2 checkpoint's score (inference/vla/infer_navsim_flow_matching_PDMS_87.2.sh
# + eval_navsim_metric_from_json.sh), but restricted to the disjoint trainval
# held-out subset (tools/rl_prep/build_eval_subset_pkl.py) instead of the full
# navtest split — see docs/rl_world_model_design.md 阶段 2.
#
# Run once per checkpoint you want to compare. EVAL_EMU_HUB must point to a
# directory that directly contains config.json + the safetensors shards —
# for a baseline checkpoint that's the checkpoint dir itself, but for an RL
# run it's a specific step_XXXXXX subdirectory under RL_CHECKPOINT_DIR
# (rl_stage1_prototype.py saves each save-point separately), NOT the
# RL_CHECKPOINT_DIR root — pointing at the root will fail (no config.json
# there). Always give each checkpoint its own EVAL_EXPERIMENT_NAME so
# outputs never collide across runs:
#   EVAL_EMU_HUB=$ASSET_ROOT/pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2 \
#     EVAL_EXPERIMENT_NAME=before_rl_trainval_eval \
#     sbatch scripts/hpc/submit_eval_subset.slurm
#   for STEP in 000100 000200 000300 000400; do
#     EVAL_EMU_HUB=$ASSET_ROOT/pretrained_models/DriveVLA-W0/<RL_CHECKPOINT_DIR>/step_${STEP} \
#       EVAL_EXPERIMENT_NAME=<label>_step_${STEP} \
#       sbatch scripts/hpc/submit_eval_subset.slurm
#   done
# Always evaluate the *whole* checkpoint curve (every save-point), not just
# the final step — 阶段 2/3 both found the interesting behavior mid-training,
# not at the end.
#
# Prerequisite (one-time): build the held-out eval subset —
#   RL_CACHE_TARGET=eval python tools/rl_prep/cache_trainval_rl_scenes.py   (navsim env)
#   python tools/rl_prep/build_eval_subset_pkl.py                          (drivevla env)

#SBATCH --job-name=eval_subset
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# Stage B (run_pdm_score.py) uses Ray to score proposals across many parallel
# workers, each loading its own metric-cache/simulator state — much more
# memory-hungry than the metric-caching pass; 32g OOM-killed it partway
# through the 286-scene "Ray objects" phase, bumped generously.
#SBATCH --mem=96g
#SBATCH --time=04:00:00
# Set notification email at submission time if desired.
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail

# A unified checkpoint directory contains the standard DriveVLA weights plus
# the jointly trained correction policy and its frozen WM outcome environment.
if [[ -n "${EVAL_UNIFIED_POLICY_DIR:-}" ]]; then
    export EVAL_EMU_HUB="$EVAL_UNIFIED_POLICY_DIR"
    export EVAL_UNIFIED_CORRECTION_CHECKPOINT="$EVAL_UNIFIED_POLICY_DIR/unified_correction_policy.pt"
    export EVAL_UNIFIED_OUTCOME_CHECKPOINT="$EVAL_UNIFIED_POLICY_DIR/unified_outcome_critic.pt"
fi

: "${EVAL_EMU_HUB:?must set EVAL_EMU_HUB to the checkpoint dir to evaluate}"
: "${EVAL_EXPERIMENT_NAME:?must set EVAL_EXPERIMENT_NAME (e.g. before_rl_trainval_eval / after_rl_trainval_eval)}"

REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
ASSET_ROOT="${DRIVEVLA_ASSET_ROOT:-$REPO_ROOT}"

module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$REPO_ROOT"

nvidia-smi --query-gpu=name,memory.total --format=csv

# ---------------------------------------------------------------------------
# Stage A (drivevla env): inference on the held-out subset -> per-token JSON
# ---------------------------------------------------------------------------
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"

export VLA_ACTION_TOKENIZER="$ASSET_ROOT/pretrained_models/fast"
export VLA_VLM_MODEL="$ASSET_ROOT/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json"
# Remaps the original authors' absolute VQ-code paths (baked into the pkl's
# image-path strings) to where this repo actually keeps them locally — same
# mapping submit_rl_stage1.slurm exports, needed by utils/datasets.py for any
# script that loads VQ-coded images, not just training.
export VLA_PATH_REPLACEMENTS="/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:$ASSET_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:$ASSET_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:$ASSET_ROOT/data/navsim/processed_data/data/navsim/processed_data/"
# utils/datasets.py holds the bare `datasets` module imported by
# utils/train_pi0.py (`from datasets import Emu3SFTDataset`) — needs
# $REPO_ROOT/utils itself on path, not just $REPO_ROOT (config.py's
# setup_paths_early() only adds train/ and reference/Emu3/, not this one).
export PYTHONPATH="$REPO_ROOT/inference/navsim/navsim:$REPO_ROOT/utils:$REPO_ROOT:${PYTHONPATH:-}"

STAGE_A_OUTPUT="$REPO_ROOT/output/${EVAL_EXPERIMENT_NAME}"
SCORE_INPUT="$STAGE_A_OUTPUT"

# Three-level decomposition (tau^0 / tau^ref / tau^1 all real-NAVSIM-scored
# from the *same* generated proposals, not cross-run/cross-checkpoint
# approximations): EVAL_REUSE_STAGE_A_FROM points at a Stage A output
# directory a prior invocation of this script already produced for the same
# checkpoint+seed, and this run scores a different downstream stage
# (raw/projection-only/full-correction) against those exact same tau^0
# proposals instead of re-running inference (and re-sampling flow-matching
# noise) a second/third time.
if [[ -n "${EVAL_REUSE_STAGE_A_FROM:-}" ]]; then
    echo "=== [1/2] Reusing Stage A output from $EVAL_REUSE_STAGE_A_FROM (skipping inference) ==="
    STAGE_A_OUTPUT="$EVAL_REUSE_STAGE_A_FROM"
    SCORE_INPUT="$STAGE_A_OUTPUT"
else
    echo "=== [1/2] Stage A inference (drivevla env), checkpoint=$EVAL_EMU_HUB ==="
    # torchrun默认绑定 29500 端口做 rendezvous —— 多个 eval job 如果被 SLURM
    # 分到同一个物理节点(不同 GPU,共享网络命名空间)会互相撞端口
    # (EADDRINUSE)。用 job ID 派生一个各自独立的端口。
    MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
    INFERENCE_ARGS=( \
        --emu_hub "$EVAL_EMU_HUB" \
        --output_dir "$STAGE_A_OUTPUT" \
        --test_data_pkl "${EVAL_TEST_DATA_PKL:-$ASSET_ROOT/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_rl_eval_pre_1s.pkl}" \
        --token_yaml "${EVAL_TOKEN_YAML:-$ASSET_ROOT/data/navsim/processed_data/scene_files/scene_filter/trainval_rl_eval.yaml}" \
        --seed "${EVAL_SEED:-0}" \
        # Always exported, not just when unified correction is set: makes
        # every Stage A output directory reusable later via
        # EVAL_REUSE_STAGE_A_FROM for a unified-correction downstream stage,
        # without having to predict at generation time whether it will be.
        --export_vlm_context \
    )
    if [[ -n "${EVAL_NATIVE_CONTROL_CHECKPOINT:-}" ]]; then
        echo "=== Native control Action Expert: $EVAL_NATIVE_CONTROL_CHECKPOINT ==="
        INFERENCE_ARGS+=(--native_control_checkpoint "$EVAL_NATIVE_CONTROL_CHECKPOINT")
    fi
    torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
        inference/vla/inference_action_navsim_flow_matching_vava.py "${INFERENCE_ARGS[@]}"
fi

# Unified mainline: one learned correction policy receives the DriveVLA
# proposal, WM outcome context, and analytic physics, then emits bounded
# residual controls rolled through the differentiable vehicle model.
if [[ -n "${EVAL_UNIFIED_CORRECTION_CHECKPOINT:-}" ]]; then
    if [[ -n "${EVAL_CONTROL_ADAPTER_CHECKPOINT:-}" ]]; then
        echo "Set either EVAL_UNIFIED_CORRECTION_CHECKPOINT or EVAL_CONTROL_ADAPTER_CHECKPOINT, not both" >&2
        exit 2
    fi
    : "${EVAL_UNIFIED_OUTCOME_CHECKPOINT:?required with unified correction}"
    UNIFIED_OUTPUT="$REPO_ROOT/output/${EVAL_EXPERIMENT_NAME}_unified_correction"
    echo "=== [1.5/2] Applying unified correction policy: $EVAL_UNIFIED_CORRECTION_CHECKPOINT ==="
    UNIFIED_ARGS=( \
        --input-dir "$STAGE_A_OUTPUT" \
        --output-dir "$UNIFIED_OUTPUT" \
        --policy-checkpoint "$EVAL_UNIFIED_CORRECTION_CHECKPOINT" \
        --outcome-checkpoint "$EVAL_UNIFIED_OUTCOME_CHECKPOINT" \
        --tokens-json "${EVAL_TOKENS_JSON:-$ASSET_ROOT/data/navsim/processed_data/anchor/trainval_rl_eval_tokens_cached.json}" \
        --wm-checkpoint "${EVAL_WM_CHECKPOINT:-$ASSET_ROOT/pretrained_models/DriveVLA-W0/Emu3_NuPlan_Pretrain_Cktps}" \
        --wm-scene-pkl "$ASSET_ROOT/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl" \
        --wm-fast-tokenizer "$ASSET_ROOT/pretrained_models/fast" \
        --wm-norm-stats "$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json" \
        --wm-vq-root "$ASSET_ROOT/data/navsim/processed_data/data/navsim/processed_data" \
    )
    # '+Projection' Table 1 arm: deploy tau^ref directly, no learned
    # correction. EVAL_UNIFIED_CORRECTION_CHECKPOINT still has to point at
    # some correction-policy checkpoint (any one -- e.g. a warm-start
    # snapshot) purely for its vehicle-model config/projection_iterations
    # metadata; project() never reads its trained weights.
    if [[ "${EVAL_PROJECTION_ONLY:-0}" == "1" && "${EVAL_RAW_ONLY:-0}" == "1" ]]; then
        echo "Set at most one of EVAL_PROJECTION_ONLY / EVAL_RAW_ONLY" >&2
        exit 2
    fi
    if [[ "${EVAL_PROJECTION_ONLY:-0}" == "1" ]]; then
        UNIFIED_ARGS+=(--projection-only)
    fi
    # tau^RL (raw Action Expert proposal) Table-2 decomposition arm: deploy
    # the Stage A proposal directly, no projection, no correction. Combine
    # with EVAL_REUSE_STAGE_A_FROM pointing at a prior run's Stage A output
    # for the *same* checkpoint+seed to score tau^RL against the exact same
    # proposals already scored as tau^proj, instead of paying for a second
    # inference pass.
    if [[ "${EVAL_RAW_ONLY:-0}" == "1" ]]; then
        UNIFIED_ARGS+=(--raw-only)
    fi
    python tools/rl_prep/apply_unified_correction_policy.py "${UNIFIED_ARGS[@]}"
    SCORE_INPUT="$UNIFIED_OUTPUT"
fi

# Optional waypoint -> bounded continuous control -> differentiable vehicle
# projection.  Disabled by default, so all existing evaluation commands retain
# their original behavior.  The proxy initial state is inferred only from the
# policy's own predicted waypoints; no held-out future/metric label is used.
if [[ -n "${EVAL_CONTROL_ADAPTER_CHECKPOINT:-}" ]]; then
    CONTROL_OUTPUT="$REPO_ROOT/output/${EVAL_EXPERIMENT_NAME}_control_adapter"
    echo "=== [1.5/2] Applying waypoint control adapter: $EVAL_CONTROL_ADAPTER_CHECKPOINT ==="
    CONTROL_ARGS=( \
        --input-dir "$STAGE_A_OUTPUT" \
        --output-dir "$CONTROL_OUTPUT" \
        --checkpoint "$EVAL_CONTROL_ADAPTER_CHECKPOINT" \
        --tokens-json "${EVAL_TOKENS_JSON:-$ASSET_ROOT/data/navsim/processed_data/anchor/trainval_rl_eval_tokens_cached.json}" \
        --projection-mode "${EVAL_CONTROL_ADAPTER_MODE:-all}" \
        --max-position-rmse "${EVAL_CONTROL_ADAPTER_MAX_POSITION_RMSE:-inf}" \
        --max-heading-rmse "${EVAL_CONTROL_ADAPTER_MAX_HEADING_RMSE:-inf}" \
    )
    if [[ -n "${EVAL_WM_CRITIC_CHECKPOINT:-}" ]]; then
        CONTROL_ARGS+=( \
            --wm-critic-checkpoint "$EVAL_WM_CRITIC_CHECKPOINT" \
            --wm-checkpoint "${EVAL_WM_CHECKPOINT:-$ASSET_ROOT/pretrained_models/DriveVLA-W0/Emu3_NuPlan_Pretrain_Cktps}" \
            --wm-scene-pkl "$ASSET_ROOT/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl" \
            --wm-fast-tokenizer "$ASSET_ROOT/pretrained_models/fast" \
            --wm-norm-stats "$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json" \
            --wm-vq-root "$ASSET_ROOT/data/navsim/processed_data/data/navsim/processed_data" \
            --candidate-range-scales "${EVAL_CONTROL_CANDIDATE_SCALES:-0.70,0.85,1.0}" \
            --wm-risk-weights "${EVAL_WM_RISK_WEIGHTS:-1,1,1,1}" \
            --wm-risk-coef "${EVAL_WM_RISK_COEF:-1.0}" \
            --track-coef "${EVAL_CONTROL_TRACK_COEF:-0.1}" \
        )
    fi
    python tools/rl_prep/apply_waypoint_control_adapter.py "${CONTROL_ARGS[@]}"
    SCORE_INPUT="$CONTROL_OUTPUT"
fi

# ---------------------------------------------------------------------------
# Stage B (navsim env): score those predictions against the held-out
# metric_cache, restricted to the same token subset
# ---------------------------------------------------------------------------
conda activate navsim
export PATH="$CONDA_PREFIX/bin:$PATH"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$ASSET_ROOT/navsim_data/maps"
export NAVSIM_EXP_ROOT="${POSTTRAIN_METRIC_OUTPUT:-$ASSET_ROOT/navsim_data/exp}"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT/inference/navsim/navsim"
export OPENSCENE_DATA_ROOT="$ASSET_ROOT/navsim_data"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

echo "=== [2/2] Stage B PDM scoring (navsim env) ==="
python tools/rl_prep/eval_pdm_score_subset.py \
    --experiment-path "$SCORE_INPUT" \
    --experiment-name "$EVAL_EXPERIMENT_NAME" \
    --metric-cache-path "${EVAL_METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_trainval_rl_eval}" \
    --subset-tokens-json "${EVAL_TOKENS_JSON:-$ASSET_ROOT/data/navsim/processed_data/anchor/trainval_rl_eval_tokens_cached.json}" \
    --train-test-split trainval

echo "=== 完成 ==="
