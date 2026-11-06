#!/bin/bash
# 在 HPC 上复现本机 `drivevla` conda 环境。
#
# 不要在登录节点上跑这个脚本，先用 srun 申请一个交互式 GPU 节点再执行，例如
# （--account 换成你自己的 subaccount，不是 root account；--gres 型号看
# `sacctmgr -p show assoc user=$USER format=account,partition` 和实际可用 GPU 定）：
#   srun --account=xinxia0 --partition=gpu --gres=gpu:v100:1 \
#        --cpus-per-task=8 --mem=32G --time=02:00:00 --pty /bin/bash
#
# 用法（在申请到的计算节点上）：
#   module load python3.10-anaconda/2023.03   # 具体名字用 `module avail 2>&1 | grep -i anaconda` 查
#   bash scripts/hpc/setup_drivevla_env.sh
#
# 关于 flash-attn：本仓库代码里 flash_attn 是条件导入
# （reference/Emu3/emu3/mllm/modeling_emu3.py: `if is_flash_attn_2_available(): ...`），
# 没装不会报错，所有脚本也一直显式用的是 attn_implementation="sdpa"，不依赖它。
# 默认跳过，不浪费时间编译。
#
# flash-attn 2.x 只支持 Ampere 及更新架构（SM80+：A100/RTX30系列/L40/RTX6000 Ada 等），
# V100（Volta, SM70）硬件层面就不支持，装了也用不上。如果之后换到 Ampere+ 的 GPU、
# 确实想要那点额外性能，再手动装（需要先 `module load gcc/11.2.0`，
# export CC/CXX 指过去，再 `MAX_JOBS=4 pip install flash_attn==2.8.3.post1 --no-build-isolation`，
# 且 --gres 要换成支持的 GPU 型号，MAX_JOBS 按实际内存调，编译很吃内存容易被 OOM kill）。

set -euo pipefail

ENV_NAME="drivevla"
PY_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROZEN_REQS="$SCRIPT_DIR/drivevla_frozen_requirements.txt"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! command -v conda &> /dev/null; then
    echo "conda not found — 先 module load 对应的 anaconda 模块再跑这个脚本" >&2
    exit 1
fi

echo "=== 创建 conda 环境: $ENV_NAME (python $PY_VERSION) ==="
conda create -n "$ENV_NAME" python="$PY_VERSION" -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== 确认 GPU / CUDA 可见 ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "警告：这个节点看不到 GPU"

echo "=== 装 torch ==="
pip install --no-cache-dir \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== 装其余依赖（跳过 flash-attn，见脚本头部说明）==="
grep -vE "^torch==|^torchvision==|^torchaudio==|^flash_attn==" "$FROZEN_REQS" > /tmp/drivevla_reqs_final.txt
pip install --no-cache-dir -r /tmp/drivevla_reqs_final.txt

echo "=== 验证 ==="
python -c "
import torch, transformers
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'cuda version:', torch.version.cuda)
print('transformers', transformers.__version__)
"

echo "=== 完成。用 'conda activate $ENV_NAME' 激活 ==="
