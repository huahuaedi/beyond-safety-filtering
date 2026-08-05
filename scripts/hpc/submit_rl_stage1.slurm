#!/bin/bash
# Great Lakes (SLURM) submission script for tools/rl_prep/rl_stage1_prototype.py.
# Confirmed working config as of 2026-08-02 (see docs/rl_world_model_design.md,
# 阶段 1 结果): account=xinxia0, partition=gpu-rtx6000 (RTX PRO 6000 Blackwell,
# ~96GB) — plenty of headroom for AdamW + KL + larger group sizes, unlike the
# 24GB local card or the 16GB V100s on this same cluster (--partition=gpu).

#SBATCH --job-name=drivevla_rl_stage1
#SBATCH --account=xinxia0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=12:00:00
# Set notification email at submission time if desired, for example:
# sbatch --mail-user=you@example.edu --mail-type=BEGIN,END,FAIL this_script.slurm
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
REPO_ROOT="$HOME/DriveVLA-W0"
CONDA_ENV="drivevla"

module load python3.10-anaconda/2023.03
module load cuda/12.8.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
# 保险：见 submit_cache_trainval_rl_scenes.slurm 里同样的说明——不同分区计算节点
# 镜像不一致，module 系统有时会把自己的 python 排到 conda env 前面。
export PATH="$CONDA_PREFIX/bin:$PATH"
cd "$REPO_ROOT"

nvidia-smi --query-gpu=name,memory.total --format=csv
free -h

# ---------------------------------------------------------------------------
# 数据路径（pretrained_models/data/navsim_data 在 ~/DriveVLA-W0 下是软链接，
# 实际落在 /scratch/xinxia_root/xinxia0/huamin/DriveVLA-W0-data/，脚本本身
# 不用感知这点，走的都是相对 REPO_ROOT 的路径）
# ---------------------------------------------------------------------------
export VLA_ACTION_TOKENIZER="$REPO_ROOT/pretrained_models/fast"
export VLA_VLM_MODEL="$REPO_ROOT/pretrained_models/Emu3-Stage1"
export VLA_NORM_STATS="$REPO_ROOT/configs/normalizer_navsim_trainval/norm_stats.json"
export VLA_PATH_REPLACEMENTS="/mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/vdb1/shuyao.shang/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/;/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data/:$REPO_ROOT/data/navsim/processed_data/data/navsim/processed_data/"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# 真实 NAVSIM PDM 仿真器奖励（不再是 8192-anchor 最近邻近似）。前置条件（一次性，
# 跑 scripts/hpc/submit_cache_trainval_rl_scenes.slurm 就够）：
#   - navsim conda 环境已建好 (scripts/hpc/setup_navsim_env.sh)
#   - trainval navsim_logs 已下载 (scripts/hpc/fetch_data.sh rl_navsim_scenes)
#   - data/navsim/processed_data/anchor/trainval_rl_scene_tokens_cached.json
#     和 $NAVSIM_EXP_ROOT/metric_cache_trainval_rl/ 已生成
# ---------------------------------------------------------------------------
export RL_REWARD_SOURCE=navsim
# 不用 `$(conda info --base)` 拼路径 —— module load 之后它解析到的是 HPC 模块系统
# 自带的系统 conda 装载路径（/sw/pkgs/...），不是用户自己的 conda 安装，navsim 环境
# 实际装在 ~/.conda/envs/navsim 下，踩过一次坑，直接写死更可靠。
export NAVSIM_ENV_PYTHON="$HOME/.conda/envs/navsim/bin/python"
export NAVSIM_METRIC_CACHE_PATH="$REPO_ROOT/navsim_data/exp/metric_cache_trainval_rl"

# ---------------------------------------------------------------------------
# 训练好的 checkpoint 存这里 —— 阶段 2 的训练前/后对比评测(submit_eval_subset.slurm)
# 需要真的能加载这份权重，不是只看训练日志里的 reward/kl 数字。每 RL_SAVE_EVERY 步存
# 一次独立子目录(rl_stage1_prototype.py 现在按 step_XXXXXX 分开存，不会互相覆盖)。
# ---------------------------------------------------------------------------
export RL_CHECKPOINT_DIR="$REPO_ROOT/pretrained_models/DriveVLA-W0/rl_stage1_5_checkpoint_v4"
export RL_SAVE_EVERY=100

# ---------------------------------------------------------------------------
# RL 训练超参。诊断历史(见 docs/rl_world_model_design.md 阶段 2-3):
#   run1 (1000步/KL_COEF=0.5): held-out PDMS 0.9213->0.8591，TTC 大跌换 progress 上涨。
#   run2 (400步/KL_COEF=1.5，3倍收紧): 退化在 100 步内就出现且不随步数恶化——
#     问题不在 KL 松紧，而在 reward 结构。
#   run3 (400步，navsim_score_worker.py `train_reward` 字段，TTC:progress 从
#     官方 5:5 改成 8:2 加权): 只是换了兑换比率，安全项依然能被 progress 补偿，
#     不是真正的修法。
# 这一轮换成 Safety-Gated GRPO：advantage 不再来自单一加权 reward 标量，
# 而是 rl_stage1_prototype.py 的 compute_safety_gated_advantage —— 安全约束
# (no_at_fault_collisions/drivable_area_compliance/TTC)是硬门槛，不满足的候选
# 无论 progress 多高，advantage 恒为负；安全候选内部才按 progress 做组内相对排序。
# 评测阶段依然用未改动的官方 PDMS。
# ---------------------------------------------------------------------------
export RL_GROUP_SIZE=16
export RL_NUM_STEPS=400
export RL_DENOISE_STEPS=10
export RL_KL_COEF=1.0
export RL_NOISE_STD=0.1
export RL_LR=1e-5
export RL_OPTIMIZER=adamw

srun python tools/rl_prep/rl_stage1_prototype.py
