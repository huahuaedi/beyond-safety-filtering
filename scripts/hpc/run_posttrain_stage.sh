#!/bin/bash
set -euo pipefail
cd "$DRIVEVLA_REPO_ROOT"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$DRIVEVLA_REPO_ROOT/inference/navsim/navsim:$DRIVEVLA_REPO_ROOT/utils:$DRIVEVLA_REPO_ROOT"
BASE="$DRIVEVLA_ASSET_ROOT/pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
export POSTTRAIN_METRIC_OUTPUT="$POSTTRAIN_RUN/metrics"
export RL_POLICY_CHECKPOINT="$BASE" RL_TRAIN_TOKENS="$POSTTRAIN_RUN/train_tokens.json"
export RL_GROUP_SIZE=16 RL_SEED=0 RL_LR=1e-5 RL_KL_COEF=1 RL_REWARD_SOURCE=navsim
export RL_FREEZE_ACTION_EXPERT=0 RL_COLLECT_ONLY=0 RL_RISK_WEIGHTED_SAMPLING=0 RL_ADAPTIVE_KL=0
export RL_RISK_MANIFEST='' RL_RISK_COVERAGE_MANIFEST='' RL_UNIFIED_CORRECTION_CHECKPOINT='' RL_UNIFIED_OUTCOME_CHECKPOINT='' RL_CONTROL_ADAPTER_CHECKPOINT=''
export RL_DIAGNOSTICS=1 RL_SAVE_STEPS='' RL_SAVE_EVERY=50 RL_DENOISE_STEPS=10 RL_NOISE_STD=0.1 RL_OPTIMIZER=adamw
export RL_RISK_BUFFER_SIZE=32 RL_RISK_REPLAY_EVERY=1 RL_RISK_REPLAY_COEF=1 RL_RISK_ONLINE_INSERT=1 RL_RISK_CURRICULUM=0 RL_DYNAMIC_CONSTRAINT_WEIGHTS=0
export EVAL_UNIFIED_POLICY_DIR='' EVAL_UNIFIED_CORRECTION_CHECKPOINT='' EVAL_CONTROL_ADAPTER_CHECKPOINT='' EVAL_NATIVE_CONTROL_CHECKPOINT='' EVAL_REUSE_STAGE_A_FROM=''
export EVAL_SEED=0
case "$1" in
 prepare)
 python tools/rl_prep/prepare_posttrain.py --assets "$DRIVEVLA_ASSET_ROOT" --run "$POSTTRAIN_RUN"
 PYTHONPATH=tools/rl_prep python -m unittest test_review_regressions test_policy_optimization test_adaptive_reward test_risk_curriculum test_rollout_diagnostics test_posttrain_pipeline
 ;;
 smoke|train)
 if [[ "$1" == smoke ]]; then export RL_NUM_STEPS=2 RL_CHECKPOINT_DIR="$POSTTRAIN_RUN/smoke"; else export RL_NUM_STEPS=200 RL_CHECKPOINT_DIR="$POSTTRAIN_RUN/checkpoints"; fi
 bash scripts/hpc/submit_rl_stage1.slurm
 test -f "$RL_CHECKPOINT_DIR/step_$(printf '%06d' "$RL_NUM_STEPS")_final/config.json"
 ;;
 val_baseline|val|test)
 if [[ "$1" == test ]]; then
  export EVAL_TOKENS_JSON="$POSTTRAIN_RUN/test_tokens.json" EVAL_TEST_DATA_PKL="$POSTTRAIN_RUN/test.pkl" EVAL_TOKEN_YAML="$POSTTRAIN_RUN/test.yaml"
  export EVAL_METRIC_CACHE_PATH="$DRIVEVLA_ASSET_ROOT/navsim_data/exp/metric_cache_trainval_rl_eval"
  if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then LABEL=baseline; export EVAL_EMU_HUB="$BASE"; else LABEL=selected; export EVAL_EMU_HUB="$(python -c 'import json,os; print(json.load(open(os.environ["POSTTRAIN_RUN"]+"/selected.json"))["checkpoint"])')"; fi
  export EVAL_EXPERIMENT_NAME="$(basename "$POSTTRAIN_RUN")_test_$LABEL"
 else
  export EVAL_TOKENS_JSON="$POSTTRAIN_RUN/val_tokens.json" EVAL_TEST_DATA_PKL="$POSTTRAIN_RUN/val.pkl" EVAL_TOKEN_YAML="$POSTTRAIN_RUN/val.yaml"
  export EVAL_METRIC_CACHE_PATH="$DRIVEVLA_ASSET_ROOT/navsim_data/exp/metric_cache_checkpoint_selection_val"
  if [[ "$1" == val_baseline ]]; then LABEL=baseline; export EVAL_EMU_HUB="$BASE"; else STEP=$((50*(SLURM_ARRAY_TASK_ID+1))); LABEL="$STEP"; SUFFIX=''; if [[ "$STEP" == 200 ]]; then SUFFIX=_final; fi; export EVAL_EMU_HUB="$POSTTRAIN_RUN/checkpoints/step_$(printf '%06d' "$STEP")$SUFFIX"; fi
  export EVAL_EXPERIMENT_NAME="$(basename "$POSTTRAIN_RUN")_val_$LABEL"
 fi
 bash scripts/hpc/submit_eval_subset.slurm
 ;;
 select|final) python tools/rl_prep/summarize_posttrain.py --run "$POSTTRAIN_RUN" --stage "$1" --baseline "$BASE" ;;
 *) exit 2;;
esac
