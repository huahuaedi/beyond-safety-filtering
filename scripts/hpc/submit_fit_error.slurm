#!/bin/bash
#SBATCH --job-name=fit_error
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
# Approximation error eps_w on the training scenes: base policy and the seed-20 fitted policy, each against its own winner.
# Required env: RUN_STD (dir holding scaled_v1_<rule>pref_seed20/action_policy.pt), PAIRS_STD (dir holding prefs_v3_{pdms,margin,nearest}/pairs.jsonl), FIT_OUT.
set -euo pipefail
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
cd "$ROOT"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$ROOT/tools/rl_prep:$ROOT/inference/navsim/navsim:$ROOT/utils:$ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
RUN_STD="${RUN_STD:?set RUN_STD}"; PAIRS_STD="${PAIRS_STD:?set PAIRS_STD}"; OUT="${FIT_OUT:?set FIT_OUT}"
SRC="$ROOT/navsim_data/exp/world_action_full_v1"
PY="$HOME/.conda/envs/drivevla/bin/python"
mkdir -p "$OUT"
for r in pdms margin nearest; do
  P="$PAIRS_STD/prefs_v3_$r/pairs.jsonl"
  CK="$RUN_STD/scaled_v1_${r}pref_seed20/action_policy.pt"
  [ -f "$P" ] && [ -f "$CK" ] || { echo "missing $P or $CK"; exit 1; }
  "$PY" -u tools/rl_prep/measure_fit_error.py --pairs "$P" --source-root "$SRC" --output "$OUT/base_vs_${r}.json"
  "$PY" -u tools/rl_prep/measure_fit_error.py --pairs "$P" --source-root "$SRC" --checkpoint "$CK" --output "$OUT/fit_${r}_s20.json"
done
