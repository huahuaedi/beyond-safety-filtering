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
if [[ "$1" == prepare ]]; then
 python tools/rl_prep/risk_replay_experiment.py --baseline-run "$BASELINE_RUN" --manifest "$SOURCE_MANIFEST" --output "$REPLAY_RUN"
 PYTHONPATH=tools/rl_prep python -m unittest test_posttrain_pipeline test_risk_replay_experiment test_review_regressions -v
 exit
fi
if [[ "$1" == summary ]]; then python tools/rl_prep/summarize_risk_replay.py --run "$REPLAY_RUN" --baseline-run "$BASELINE_RUN"; exit; fi
MODES=(matched_reference rescored_rl)
MODE="${MODES[$((SLURM_ARRAY_TASK_ID % 2))]}"
SEED="$((SLURM_ARRAY_TASK_ID / 2))"
ARM="${MODE}_seed${SEED}"
if [[ "$1" == eval ]]; then
 export EVAL_EMU_HUB="$REPLAY_RUN/$ARM/step_000100_final"
 export EVAL_EXPERIMENT_NAME="$(basename "$REPLAY_RUN")_$ARM"
 export EVAL_TOKENS_JSON="$BASELINE_RUN/test_tokens.json" EVAL_TEST_DATA_PKL="$BASELINE_RUN/test.pkl" EVAL_TOKEN_YAML="$BASELINE_RUN/test.yaml"
 export EVAL_METRIC_CACHE_PATH="$DRIVEVLA_ASSET_ROOT/navsim_data/exp/metric_cache_trainval_rl_eval"
 export POSTTRAIN_METRIC_OUTPUT="$REPLAY_RUN/metrics" EVAL_SEED=0
 export EVAL_UNIFIED_POLICY_DIR='' EVAL_UNIFIED_CORRECTION_CHECKPOINT='' EVAL_CONTROL_ADAPTER_CHECKPOINT='' EVAL_NATIVE_CONTROL_CHECKPOINT='' EVAL_REUSE_STAGE_A_FROM=''
 bash scripts/hpc/submit_eval_subset.slurm
 exit
fi
export RL_POLICY_CHECKPOINT="$DRIVEVLA_ASSET_ROOT/pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
export RL_TRAIN_TOKENS="$BASELINE_RUN/train_tokens.json" RL_GROUP_SIZE=16 RL_SEED="$SEED" RL_LR=1e-5 RL_KL_COEF=1
export RL_RISK_REPLAY_MODE="$MODE" RL_RISK_FIXED_SCHEDULE="$REPLAY_RUN/schedule_seed${SEED}.json"
export RL_RISK_MANIFEST="$REPLAY_RUN/fixed_manifest.json" RL_RISK_COVERAGE_MANIFEST='' RL_RISK_ONLINE_INSERT=0 RL_RISK_CURRICULUM=0
export RL_RISK_BUFFER_SIZE=32 RL_RISK_REPLAY_EVERY=1 RL_RISK_REPLAY_COEF=1 RL_DYNAMIC_CONSTRAINT_WEIGHTS=0 RL_RISK_WEIGHTED_SAMPLING=0
export RL_FREEZE_ACTION_EXPERT=0 RL_COLLECT_ONLY=0 RL_ADAPTIVE_KL=0 RL_UNIFIED_CORRECTION_CHECKPOINT='' RL_UNIFIED_OUTCOME_CHECKPOINT='' RL_CONTROL_ADAPTER_CHECKPOINT=''
export RL_DIAGNOSTICS=1 RL_SAVE_STEPS=100 RL_SAVE_EVERY=100 RL_DENOISE_STEPS=10 RL_NOISE_STD=0.1 RL_OPTIMIZER=adamw
if [[ "$1" == smoke ]]; then export RL_NUM_STEPS=2 RL_CHECKPOINT_DIR="$REPLAY_RUN/smoke_$ARM"; else export RL_NUM_STEPS=100 RL_CHECKPOINT_DIR="$REPLAY_RUN/$ARM"; fi
bash scripts/hpc/submit_rl_stage1.slurm
python - <<'PY'
import json,os
from pathlib import Path
root=Path(os.environ['RL_CHECKPOINT_DIR'])
rows=[json.loads(s) for s in (root/'replay_updates.jsonl').read_text().splitlines()]
assert len(rows)==int(os.environ['RL_NUM_STEPS'])
for row in rows:
 if row['kind']=='risk':
  assert row['train_on_group']==(row['mode']=='rescored_rl')
  assert row['rehearsal']==(row['mode']=='matched_reference')
assert rows[-1]['scored_trajectories']>=len(rows)*16
assert (root/f"step_{len(rows):06d}_final/config.json").is_file()
print('Replay execution/checkpoint checks passed',flush=True)
PY
