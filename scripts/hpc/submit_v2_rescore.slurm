#!/bin/bash
#SBATCH --job-name=v2_rescore
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
# Re-score the saved trajectories of one evaluated policy with the NAVSIM v2 scorer.
# SMOKE=1: 30-scene smoke cache and filter. Env: LABEL (e.g. sft, nearestpref_s22), TRAFFIC (non_reactive|reactive), V2_CACHE (default metric_cache_navtest_v2), NAME (experiment name).
set -euo pipefail
LABEL="${LABEL:?LABEL}"; TRAFFIC="${TRAFFIC:-non_reactive}"
V2="${NAVSIM_V2_REPO:-$HOME/navsim_v2}"
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
E=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data/navsim_data
CACHE="${V2_CACHE:-$E/exp/metric_cache_navtest_v2}"
NAME="${NAME:-v2_${TRAFFIC}_${LABEL}}"; EXTRA=()
if [ "${SMOKE:-0}" = 1 ]; then CACHE="${CACHE}_smoke"; NAME="${NAME}_smoke"; EXTRA=(train_test_split.scene_filter.max_scenes=30); fi
NPZ="$E/exp/official_navtest_trained_eval/${LABEL}__shard*/validation/*.npz"
VIEW="$HOME/openscene_navtest_view"
test -d "$CACHE" && ls $NPZ >/dev/null
echo "npz files: $(ls $NPZ | wc -l)  traffic=$TRAFFIC  cache=$CACHE"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate navsim2
export PATH="$CONDA_PREFIX/bin:$PATH"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT="$E/maps"
export OPENSCENE_DATA_ROOT="$VIEW" NAVSIM_EXP_ROOT="$E/exp" NAVSIM_DEVKIT_ROOT="$V2"
export PYTHONPATH="$V2:$ROOT/tools/rl_prep:${PYTHONPATH:-}"
cd "$V2"
python navsim/planning/script/run_pdm_score_one_stage.py \
  "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$ROOT/scripts/hpc/v2_cfg]" \
  train_test_split=navtest agent=npz_trajectory_agent "agent.npz_glob='$NPZ'" \
  traffic_agents="$TRAFFIC" experiment_name="$NAME" metric_cache_path="$CACHE" ${EXTRA[@]+"${EXTRA[@]}"}
echo V2_SCORE_DONE
