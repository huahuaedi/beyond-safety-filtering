#!/bin/bash
#SBATCH --job-name=nt_official_infer
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err
# Official DriveVLA-W0 inference (inference/vla/*.py) over the FULL navtest pickle: one json per scene with the [8,3] relative action.
# MODEL=flow   -> Emu3_Flow_Matching_Action_Expert_PDMS_87.2  (published 87.2)
# MODEL=anchor -> Emu3_Multi_Anchor_PDMS_90.3                 (published 90.2/90.3, anchor-scoring planner)
# MODEL=query  -> Emu3_Query_based_Action_Expert_PDMS_88.4    (published 88.4; the checkpoint has 256 action queries, i.e. it is the anchor variant)
# Env: MODEL (flow|anchor), INFER_OUT (new directory). Scoring is a separate step (submit_official_navtest_score.slurm).
set -euo pipefail
MODEL="${MODEL:?MODEL=flow|anchor|query}"
OUT="${INFER_OUT:?INFER_OUT}"
[ -e "$OUT" ] && { echo "refusing existing $OUT"; exit 1; }
ROOT="${OFFLINE_REPO_ROOT:-$HOME/DriveVLA-Safety-RL-offline-20260907}"
REPO_ROOT="${DRIVEVLA_REPO_ROOT:-$HOME/rl_safety_order_v1_20261008_150538_33627/repo}"
ASSET=/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data
PKL="$ASSET/pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_test_pre_1s.pkl"
TOKEN_YAML="$ROOT/inference/navsim/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
case "$MODEL" in
  flow)   HUB="$ASSET/pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2"; SCRIPT=inference_action_navsim_flow_matching_vava.py ;;
  anchor) HUB="$ASSET/pretrained_models/DriveVLA-W0/Emu3_Multi_Anchor_PDMS_90.3";                SCRIPT=inference_action_navsim_query_based_vava.py ;;
  query)  HUB="$ASSET/pretrained_models/DriveVLA-W0/Emu3_Query_based_Action_Expert_PDMS_88.4";   SCRIPT=inference_action_navsim_query_based_vava.py ;;
  *) echo "unknown MODEL $MODEL"; exit 2 ;;
esac
test -f "$HUB/config.json" && test -f "$PKL" && test -f "$TOKEN_YAML"
module load python3.10-anaconda/2023.03
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"
export DRIVEVLA_ROOT="$REPO_ROOT" DRIVEVLA_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/inference/navsim/navsim:$REPO_ROOT/utils:$REPO_ROOT:${PYTHONPATH:-}"
export VLA_ACTION_TOKENIZER="$REPO_ROOT/pretrained_models/fast" VLA_VLM_MODEL="$REPO_ROOT/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json"
export VLA_TOKEN_YAML="$TOKEN_YAML" VLA_NUM_WORKERS=8 VLA_BATCH_SIZE=1
if [ "$MODEL" = anchor ] || [ "$MODEL" = query ]; then
  export VLA_ANCHOR_ENABLED=true VLA_ANCHOR_CLUSTER_PATH="$REPO_ROOT/data/navsim/processed_data/anchor/cluster_centers_8192.npy"
  export VLA_ANCHOR_METRIC_SCORE_PATH="$REPO_ROOT/data/navsim/processed_data/anchor/formatted_pdm_score_8192.npy"
fi
export VLA_PATH_REPLACEMENTS="/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The inference script names outputs token_list[dataset_index]; the pickle and the navtest token list must align one to one.
python - "$PKL" "$TOKEN_YAML" <<'PY'
import pickle, sys, yaml
pkl = pickle.load(open(sys.argv[1], "rb"))
tokens = yaml.safe_load(open(sys.argv[2]))["tokens"]
n = len(pkl["data"]) if isinstance(pkl, dict) and "data" in pkl else len(pkl)
print(f"pickle scenes={n} navtest tokens={len(tokens)} unique={len(set(tokens))}")
if n != len(tokens):
    raise SystemExit(f"pickle ({n}) and navtest token list ({len(tokens)}) differ: names would be wrong")
PY
# Jobs sharing a node would collide on torchrun's default port 29500; derive a distinct one from the job id.
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
echo "model=$MODEL hub=$HUB out=$OUT" port=$PORT
if [ "$MODEL" = flow ]; then
  torchrun --nproc_per_node=1 --master_port "$PORT" "$REPO_ROOT/inference/vla/$SCRIPT" --emu_hub "$HUB" --output_dir "$OUT" --test_data_pkl "$PKL"
else
  torchrun --nproc_per_node=1 --master_port "$PORT" "$ROOT/scripts/hpc/run_inference_anchor_patch.py" "$REPO_ROOT/inference/vla/$SCRIPT" --emu_hub "$HUB" --output_dir "$OUT" --test_data_pkl "$PKL"
fi
echo "json files: $(ls "$OUT"/*.json 2>/dev/null | wc -l)"
echo INFER_DONE
