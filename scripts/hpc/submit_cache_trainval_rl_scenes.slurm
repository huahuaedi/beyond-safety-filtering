#!/bin/bash
# 一次性准备工作：为真实 NAVSIM PDM 仿真器奖励(RL_REWARD_SOURCE=navsim)生成可用的
# trainval 训练场景池 + 一份完全不相交的 held-out 评测场景池（阶段 2 训练前后对比
# 用，见 docs/rl_world_model_design.md）+ 两者各自的 metric_cache。纯 CPU 任务，
# 不占 GPU 节点，跑一次就够，之后 submit_rl_stage1.slurm / submit_eval_subset.slurm
# 直接读它的产物。
#
# 前置条件：
#   bash scripts/hpc/fetch_data.sh rl_navsim_scenes   # 下载 trainval navsim_logs
#   bash scripts/hpc/setup_navsim_env.sh              # 建好 navsim conda 环境
# （drivevla 环境应该已经存在，见 setup_drivevla_env.sh）

#SBATCH --job-name=cache_trainval_rl_scenes
#SBATCH --account=xinxia0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=06:00:00
# Set notification email at submission time if desired.
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail

REPO_ROOT="$HOME/DriveVLA-W0"

module load python3.10-anaconda/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$REPO_ROOT"

export VLA_ACTION_TOKENIZER="$REPO_ROOT/pretrained_models/fast"
export VLA_VLM_MODEL="$REPO_ROOT/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json"

# ---------------------------------------------------------------------------
# 第一步（drivevla env）：采样候选 token 池 —— DriveVLA-W0 自己的采样网格和 NAVSIM
# 官方场景网格没有严格对齐，只有一小部分候选能落在合法的 NAVSIM 场景锚点上（第一轮
# 实测约 9.3% 命中率，28/300），所以这里先过采样，交给下一步筛出真正可用的。
# 训练池和 eval 池取自同一次 shuffle 的不同切片，保证不相交（见脚本内注释）。
# ---------------------------------------------------------------------------
echo "=== [1/3] 采样候选 trainval 场景 token 池（训练 + held-out 评测，drivevla env）==="
conda activate drivevla
# 非交互式 sbatch job 里，module 系统有时会把自己的 python/site-packages 排在
# conda env 的 bin 前面，即使 activate 之后也盖不掉（同一台集群上不同分区的计算
# 节点镜像不一致，标准分区上踩到了这个问题，GPU 分区暂时没踩到但保留这个保险）。
export PATH="$CONDA_PREFIX/bin:$PATH"
export RL_SCENE_POOL_SIZE=3000
export RL_EVAL_POOL_SIZE=4000
python tools/rl_prep/sample_trainval_scene_pool.py

# ---------------------------------------------------------------------------
# 第二步（navsim env）：用 NAVSIM 官方 SceneLoader 分别筛出训练池和 eval 池里真正
# 合法的场景锚点，各自建 metric_cache（其余候选直接跳过，不报错）。产物：
#   data/navsim/processed_data/anchor/trainval_rl_scene_tokens_cached.json
#   data/navsim/processed_data/anchor/trainval_rl_eval_tokens_cached.json
#   $NAVSIM_EXP_ROOT/metric_cache_trainval_rl/
#   $NAVSIM_EXP_ROOT/metric_cache_trainval_rl_eval/
# ---------------------------------------------------------------------------
echo "=== [2/3] 构建训练池 + eval 池的 metric_cache (navsim env) ==="
conda activate navsim
export PATH="$CONDA_PREFIX/bin:$PATH"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$REPO_ROOT/navsim_data/maps"
export NAVSIM_EXP_ROOT="$REPO_ROOT/navsim_data/exp"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT/inference/navsim/navsim"
export OPENSCENE_DATA_ROOT="$REPO_ROOT/navsim_data"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

RL_CACHE_TARGET=train python tools/rl_prep/cache_trainval_rl_scenes.py
RL_CACHE_TARGET=eval python tools/rl_prep/cache_trainval_rl_scenes.py

# ---------------------------------------------------------------------------
# 第三步（drivevla env）：把最终确认可用的 eval token 集合，过滤成一份可以直接喂给
# 推理脚本的 pkl + token yaml（trainval pkl 每条自带 token 字段，不需要像 navtest
# 那样依赖一份缺失的官方顺序映射文件）。产物：
#   pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_rl_eval_pre_1s.pkl
#   data/navsim/processed_data/scene_files/scene_filter/trainval_rl_eval.yaml
# ---------------------------------------------------------------------------
echo "=== [3/3] 构建 eval 子集 pkl + token yaml (drivevla env) ==="
conda activate drivevla
export PATH="$CONDA_PREFIX/bin:$PATH"
python tools/rl_prep/build_eval_subset_pkl.py

echo "=== 完成 ==="
