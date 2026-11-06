#!/bin/bash
# Can the representation the POLICY itself sees tell which of its samples are safe?  Same candidates, labels, selector and log-disjoint
# split as the geometry ablations; only the information given to the ranker changes:
#   trajectory only | trajectory + policy context | trajectory + map | trajectory + map + policy context | full geometry (+ policy context)
# Steps: (1) policy context of the 7,171 pool scenes and the 12,146 navtest scenes (GPU, 8 shards each), (2) PCA fitted on the pool and
# per-candidate blocks (CPU), (3) selector fit and application with --split-geometry --split-extras (CPU).
# Usage: SEL_OUT=<full selector dir> CHAIN_BASE_CSV=<official flow csv> bash scripts/hpc/run_policy_visual_test.sh
set -euo pipefail
OFF="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
X="$ASSET/navsim_data/exp"
O="${SEL_OUT:?set SEL_OUT}"
COH="${NT_COHORT:-$X/official_navtest_full_cohort2}"
EXP="$OFF/navsim_data/exp"
CSV="${CHAIN_BASE_CSV:?set CHAIN_BASE_CSV}"
POOLPKL="$ASSET/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
test -d "$O/collection" && test -d "$EXP/world_action_full_v1" && test -f "$POOLPKL" && test -f "$COH/test.pkl"
[ -e "$O/pv" ] && { echo "refusing existing $O/pv"; exit 1; }
mkdir -p "$O/pv"; cd "$OFF"
python3 - "$EXP/world_action_full_v1" "$O/pv/pool_tokens.json" <<'PY'
import json, sys
from pathlib import Path
toks = sorted(p.stem for s in ("train", "val", "test") for p in (Path(sys.argv[1]) / s).glob("*.npz"))
Path(sys.argv[2]).write_text(json.dumps(toks))
print(len(toks), "pool scenes")
PY
JP=$(PV_PKL="$POOLPKL" PV_TOKENS="$O/pv/pool_tokens.json" PV_OUT="$O/pv/pool_context" sbatch --parsable scripts/hpc/submit_policy_vlm_context.slurm)
JN=$(PV_PKL="$COH/test.pkl" PV_TOKENS="$COH/test_tokens.json" PV_OUT="$O/pv/navtest_context" sbatch --parsable scripts/hpc/submit_policy_vlm_context.slurm)
echo "context extraction: pool $JP, navtest $JN"
JB=$(sbatch --parsable --dependency=afterok:$JP:$JN --job-name=pv_blocks --account=xinxia0 --partition=standard --cpus-per-task=4 --mem=64g --time=02:00:00 \
     --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
     --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $OFF/tools/rl_prep && python make_policy_visual_blocks.py --contexts $O/pv/pool_context --candidate-root $EXP/world_action_full_v1 --out $O/pv/pool_blocks --fit && python make_policy_visual_blocks.py --contexts $O/pv/navtest_context --candidate-root $O/collection --out $O/pv/navtest_blocks --pca $O/pv/pool_blocks/pca.npz")
JF=$(sbatch --parsable --dependency=afterok:$JB --job-name=pv_select --account=xinxia0 --partition=standard --cpus-per-task=8 --mem=96g --time=04:00:00 \
     --output=$HOME/%x-%j.log --error=$HOME/%x-%j.err \
     --wrap "module load python3.10-anaconda/2023.03; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate drivevla; cd $OFF/tools/rl_prep && python fit_and_apply_selector.py --train-root $EXP/world_action_full_v1 --train-geom-root $EXP/world_action_geom_full_v1 --train-extra-root pv=$O/pv/pool_blocks --apply-root $O/collection --apply-geom-root $O/geom --apply-extra-root pv=$O/pv/navtest_blocks --token-log-map $COH/token_log_map.json --sft-csv $CSV --split-geometry --split-extras --output $O/selector_navtest_pv.json")
echo "blocks $JB, selector $JF   result $O/selector_navtest_pv.json"
