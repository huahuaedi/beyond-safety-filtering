#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
[[ ! -e config.json ]] || { echo 'Already submitted; refusing overwrite'; exit 1; }
python3 - <<'PY'
import json
from pathlib import Path
pkg=Path.cwd();assets=Path('/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data')
posttrain=assets/'pretrained_models/DriveVLA-W0/posttrain_runs'
run=posttrain/pkg.name
c=dict(repo='/home/huamin/rl_pareto_20261001_182743_81761/repo',assets=str(assets),run=str(run),
       cohort_run=str(posttrain/'rl_pareto_navtest_v2_20261005_133703_37354'))
if run.exists():raise ValueError('Output already exists')
Path('config.json').write_text(json.dumps(c,indent=2));print('run='+str(run))
PY
N=$(python3 -c 'import arms;print(len(arms.arms()))')
submit_job() { sbatch --parsable --account=xinxia0 --nodes=1 --ntasks=1 --kill-on-invalid-dep=yes --output="$PWD/%x-%A_%a.log" --error="$PWD/%x-%A_%a.err" "$@"; }
p=$(submit_job --partition=standard --cpus-per-task=4 --mem=64g --time=01:00:00 --job-name=of_prepare job.sh prepare);p=${p%%;*};echo "prepare=$p" | tee jobs.txt
s=$(submit_job --partition=gpu-rtx6000 --gres=gpu:rtx_pro_6000_blackwell:1 --cpus-per-task=16 --mem=96g --time=02:00:00 --job-name=of_smoke --dependency="afterok:$p" job.sh smoke);s=${s%%;*};echo "smoke=$s" | tee -a jobs.txt
e=$(submit_job --partition=gpu-rtx6000 --gres=gpu:rtx_pro_6000_blackwell:1 --cpus-per-task=16 --mem=96g --time=04:00:00 --array=0-$((N-1))%5 --job-name=of_eval --dependency="afterok:$s" job.sh eval);e=${e%%;*};echo "eval=$e" | tee -a jobs.txt
# report needs only the primary arms and the SFT re-evaluations; descriptive arms may fail without blocking it
r=$(submit_job --partition=standard --cpus-per-task=2 --mem=8g --time=00:20:00 --job-name=of_report --dependency="afterany:$e" job.sh report);echo "report=${r%%;*}" | tee -a jobs.txt
