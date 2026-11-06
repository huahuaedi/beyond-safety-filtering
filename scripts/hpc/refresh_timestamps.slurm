#!/bin/bash
#SBATCH --job-name=touch_inuse
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
# Refresh access/modification times of everything the current pipeline reads, so the cluster's age-based purge does not
# remove it (many of these files are tiny but indispensable: tokenizers, map tiles, cache metadata). Content is untouched;
# touch -c never creates files. Early RL checkpoints (v3_*, v4_*, reference_anchored_*, rl_stage1_5_*, v5_*) are deliberately
# NOT listed: they are verified in the backup /nfs/turbo/umd-xinxia/huamin_backup/purge_20261006_job_64322695.
set -euo pipefail
A=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
P=$A/pretrained_models
DIRS=(
  "$A/data/navsim/processed_data"
  "$P/Emu3-Stage1" "$P/Emu3-VisionTokenizer" "$P/fast"
  "$P/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
  "$P/DriveVLA-W0/Emu3_Multi_Anchor_PDMS_90.3"
  "$P/DriveVLA-W0/Emu3_Query_based_Action_Expert_PDMS_88.4"
  "$P/DriveVLA-W0/Emu3_NuPlan_Pretrain_Cktps"
  "$P/DriveVLA-W0/posttrain_runs/rl_pareto_navtest_v2_20261005_133703_37354"
  "$A/navsim_data/maps" "$A/navsim_data/navsim_logs"
)
for d in "$A"/navsim_data/exp/metric_cache_* "$A"/navsim_data/exp/world_action*; do [ -d "$d" ] && DIRS+=("$d"); done
for d in "${DIRS[@]}"; do
  [ -d "$d" ] || { echo "skip (missing) $d"; continue; }
  n=$(find "$d" -type f -exec touch -c {} + -print | wc -l || true)
  echo "touched $n files under $d"
done
find "$P/DriveVLA-W0" -maxdepth 1 -type f -exec touch -c {} + || true   # the released pickles
echo TOUCH_DONE
