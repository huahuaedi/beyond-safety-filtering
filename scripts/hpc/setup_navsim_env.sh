#!/bin/bash
# 在 HPC 上复现本机 `navsim` conda 环境（跑真实 NAVSIM PDM 仿真器打分用，独立于
# `drivevla` 环境 —— nuplan-devkit 需要 numpy==1.23.4 / torch==2.0.1 / python 3.9，
# 和 drivevla 的 torch 2.11 + numpy 2.x 不兼容，两边只能通过子进程 + 文件 I/O 通信，
# 见 tools/rl_prep/navsim_score_worker.py）。
#
# 不要在登录节点上跑：这一步要 clone/build nuplan-devkit，比较吃 CPU/网络，用
# CPU 分区的交互节点就够，不需要 GPU：
#   srun --account=xinxia0 --partition=standard --cpus-per-task=8 --mem=32G \
#        --time=02:00:00 --pty /bin/bash
#
# 用法（在申请到的节点上）：
#   module load python3.10-anaconda/2023.03
#   bash scripts/hpc/setup_navsim_env.sh

set -euo pipefail

ENV_NAME="navsim"
PY_VERSION="3.9"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAVSIM_PKG_DIR="$REPO_ROOT/inference/navsim/navsim"

if ! command -v conda &> /dev/null; then
    echo "conda not found — 先 module load 对应的 anaconda 模块再跑这个脚本" >&2
    exit 1
fi
if [ -n "${CONDA_PREFIX:-}" ]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi

echo "=== 创建 conda 环境: $ENV_NAME (python $PY_VERSION) ==="
conda create -n "$ENV_NAME" python="$PY_VERSION" pip=23.3.1 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== 装依赖（含 nuplan-devkit，从 gitee 拉源码装，会比较慢，正常）==="
pip install --no-cache-dir -r "$NAVSIM_PKG_DIR/requirements.txt"

echo "=== 把 navsim 包本身以可编辑模式装上 ==="
pip install --no-cache-dir -e "$NAVSIM_PKG_DIR"

echo "=== 验证 ==="
python -c "
import numpy, torch, navsim
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
print('numpy', numpy.__version__, 'torch', torch.__version__)
print('navsim + PDMSimulator import OK')
"

echo "=== 完成。用 'conda activate $ENV_NAME' 激活 ==="
