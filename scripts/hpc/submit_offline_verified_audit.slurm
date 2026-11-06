#!/bin/bash
#SBATCH --job-name=offline_verified_audit
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
cd "$ROOT"
export PYTHONPATH="$ROOT/tools/rl_prep:$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY="$HOME/.conda/envs/drivevla/bin/python"
"$PY" -m unittest discover -s tools/rl_prep -p test_offline_verified_data.py -v
"$PY" tools/rl_prep/audit_offline_verified_data.py \
 --root "$ROOT/navsim_data/exp/world_action_full_v1" \
 --scene-pkl "$ROOT/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl" \
 --eval-tokens "$ROOT/data/navsim/processed_data/anchor/trainval_rl_eval_tokens_cached.json" \
 --val-tokens "$ROOT/data/navsim/processed_data/anchor/checkpoint_selection_val_tokens_cached.json" \
 --output "$ROOT/output/offline_verified_20260907/audit.json"
