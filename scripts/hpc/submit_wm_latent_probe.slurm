#!/bin/bash
# CPU-only, memory-heavy offline analysis (world-model latent probes).
# No GPU actually needed by the workload, but this cluster's job
# accounting is organized around the gpu-rtx6000 partition throughout this
# project -- reusing it here keeps queueing simple; --gres is omitted so it
# doesn't reserve a GPU.

#SBATCH --job-name=wm_latent_probe
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180g
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail
REPO_ROOT="$HOME/DriveVLA-W0"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
cd "$REPO_ROOT"

mkdir -p results_summary/wm_probe
python3 tools/rl_prep/wm_latent_probe_analysis.py --out-dir results_summary/wm_probe
