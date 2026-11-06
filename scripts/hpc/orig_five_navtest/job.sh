#!/bin/bash -l
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
while IFS= read -r key; do case "$key" in RL_*|EVAL_*|NAVSIM_*|VLA_*|POSTTRAIN_*) unset "$key";; esac; done < <(compgen -e)
read_config() { python3 -c 'import json,sys;print(json.load(open("config.json"))[sys.argv[1]])' "$1"; }
repo=$(read_config repo); assets=$(read_config assets); run=$(read_config run)
export DRIVEVLA_REPO_ROOT="$repo" DRIVEVLA_ASSET_ROOT="$assets"
export NAVSIM_DEVKIT_ROOT="$repo/inference/navsim/navsim" OPENSCENE_DATA_ROOT="$assets/navsim_data" NAVSIM_EXP_ROOT="$assets/navsim_data/exp"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT="$assets/navsim_data/maps"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$repo/utils:$repo" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$1" in
 prepare)
 conda activate navsim
 python -m unittest test_cohort -v
 python -m unittest test_orig_package -v
 python prepare.py
 ;;
 smoke|eval)
 test -f "$run/ready.json"
 id=${SLURM_ARRAY_TASK_ID:-0}
 if [[ "$1" == smoke ]]; then id=0; fi
 IFS='|' read -r arm checkpoint evalseed < <(python3 arms.py "$assets/pretrained_models/DriveVLA-W0" "$id")
 if [[ "$1" == smoke ]]; then arm=smoke_sft; fi
 echo "arm=$arm checkpoint=$checkpoint eval_seed=$evalseed"
 export EVAL_EMU_HUB="$checkpoint" EVAL_EXPERIMENT_NAME="$(basename "$run")_$arm" EVAL_SEED="$evalseed"
 export EVAL_TEST_DATA_PKL="$run/test.pkl" EVAL_TOKEN_YAML="$run/test.yaml" EVAL_TOKENS_JSON="$run/test_tokens.json"
 export EVAL_METRIC_CACHE_PATH="$run/cache" POSTTRAIN_METRIC_OUTPUT="$run/metrics"
 if [[ "$1" == smoke ]]; then
  export EVAL_TEST_DATA_PKL="$run/smoke.pkl" EVAL_TOKEN_YAML="$run/smoke.yaml" EVAL_TOKENS_JSON="$run/smoke_tokens.json"
 fi
 export EVAL_OPENSCENE_DATA_ROOT="$run/navsim_data"
 export NAVTEST_EVAL_PACKAGE="$SLURM_SUBMIT_DIR"
 bash "$SLURM_SUBMIT_DIR/eval.sh"
 if [[ "$1" == smoke ]]; then conda activate drivevla; python check_smoke.py; fi
 ;;
 report) conda activate drivevla; python report.py ;;
 *) exit 2 ;;
esac
