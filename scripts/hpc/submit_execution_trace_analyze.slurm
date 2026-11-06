#!/bin/bash
#SBATCH --job-name=execution_trace_analyze
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
set -euo pipefail
: "${TRACE_RUN_ROOT:?}" "${TRACE_SUMMARY_DIR:?}"
cd "${DRIVEVLA_REPO_ROOT:-$HOME/DriveVLA-W0}"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/rl_prep/analyze_execution_traces.py \
    --trace-dir "$TRACE_RUN_ROOT/traces" --output-dir "$TRACE_SUMMARY_DIR"
