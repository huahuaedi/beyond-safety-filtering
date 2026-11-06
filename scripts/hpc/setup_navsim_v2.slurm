#!/bin/bash
#SBATCH --job-name=setup_navsim2
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
set -euo pipefail
bash "${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}/scripts/hpc/setup_navsim_v2.sh"
