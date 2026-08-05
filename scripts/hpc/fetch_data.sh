#!/bin/bash
# 在 HPC 上重新拉取本机验证过能跑通的模型权重和数据集，替代直接 rsync 132GB。
# 分成几个独立模块，按需要跑，不是每次都要全跑。
#
# 用法：
#   cd DriveVLA-W0
#   bash scripts/hpc/fetch_data.sh core        # 87.2 flow-matching 推理必需的最小集合
#   bash scripts/hpc/fetch_data.sh navsim_eval  # + PDMS 评测需要的 navsim/nuplan 数据
#   bash scripts/hpc/fetch_data.sh rl_anchor    # + RL 阶段1原型需要的锚点分数(不下47G原始文件)
#   bash scripts/hpc/fetch_data.sh rl_navsim_scenes  # + 真实 NAVSIM PDM 仿真器奖励需要的 trainval navsim_logs
#   bash scripts/hpc/fetch_data.sh nuplan_wm    # + 阶段0.5用到的 NuPlan world model checkpoint(~17G)
#   bash scripts/hpc/fetch_data.sh all          # 全部

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 有些 HPC 的 `module load` 会把系统自带的 python/huggingface-cli 排在
# PATH 里比 conda env 更靠前，即使 `conda activate` 之后也盖不掉，导致
# 命令解析到错误版本（比如老版本 huggingface-cli 没有 `download` 子命令）。
# 强制把当前激活的 conda env 的 bin 目录排到最前面。
if [ -n "${CONDA_PREFIX:-}" ]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi

MODE="${1:-core}"

download_core() {
    echo "=== [core] Emu3-Stage1 / Emu3-VisionTokenizer / fast / 87.2 flow-matching checkpoint / test pkl ==="
    pip install -q huggingface_hub
    mkdir -p pretrained_models
    bash scripts/misc/download_emu3_pretrain.sh  # Emu3-Stage1 / Emu3-VisionTokenizer / fast

    # download_emu3_pretrain.sh 不含实际推理用的 87.2 checkpoint 本体，单独补上。
    # 注意：不要用 --include "Emu3_Flow_Matching_Action_Expert_PDMS_87.2/*" 这种通配符，
    # 会把作者自己评测留下的 json_output/ 子目录（近万个小文件）也一起匹配进来，
    # 把匿名下载的速率限制打满(429)。明确列出真正需要的权重文件。
    mkdir -p pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2
    cd pretrained_models/DriveVLA-W0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2
    for f in config.json emu3.tiktoken emu3_vision_tokens.txt generation_config.json \
             model.safetensors.index.json model-00001-of-00004.safetensors \
             model-00002-of-00004.safetensors model-00003-of-00004.safetensors \
             model-00004-of-00004.safetensors; do
        curl -L --fail -o "$f" "https://huggingface.co/liyingyan/DriveVLA-W0/resolve/main/Emu3_Flow_Matching_Action_Expert_PDMS_87.2/$f"
    done
    cd -

    curl -L --fail -o pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_test_pre_1s.pkl \
        "https://huggingface.co/liyingyan/DriveVLA-W0/resolve/main/navsim_emu_vla_256_144_test_pre_1s.pkl"

    mkdir -p data/navsim/processed_data
    curl -L --fail -o data/navsim/processed_data/test_vq_codes.zip \
        "https://huggingface.co/liyingyan/DriveVLA-W0/resolve/main/test_vq_codes.zip"
    (cd data/navsim/processed_data && unzip -q -o test_vq_codes.zip && rm test_vq_codes.zip)
}

download_navsim_eval() {
    echo "=== [navsim_eval] nuPlan maps + NAVSIM test split metadata ==="
    mkdir -p navsim_data/maps navsim_data/navsim_logs navsim_data/sensor_blobs/test navsim_data/exp
    (cd navsim_data && \
        curl -L --fail -o nuplan-maps-v1.1.zip "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip" && \
        unzip -q -o nuplan-maps-v1.1.zip && rm -rf maps && mv nuplan-maps-v1.0 maps && rm nuplan-maps-v1.1.zip)
    (cd navsim_data && \
        curl -L --fail -o openscene_metadata_test.tgz "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz" && \
        tar -xzf openscene_metadata_test.tgz && \
        rm -rf navsim_logs/test && mkdir -p navsim_logs && mv openscene-v1.1/meta_datas navsim_logs/test && \
        rm -rf openscene-v1.1 openscene_metadata_test.tgz)
    echo "注意：navsim/nuplan-devkit conda 环境（不同于 drivevla）需要单独装，参考主对话里
    inference/navsim/navsim/environment.yml + requirements.txt，或问我要一份对应的 hpc 安装脚本。"
    echo "地图+日志装好后，用 inference/navsim/navsim/navsim/planning/script/run_metric_caching.py
    重新生成 metric_cache（CPU 任务，不需要占着 GPU 节点跑，可以用 --partition=standard 单独提交）。"
}

download_rl_anchor() {
    echo "=== [rl_anchor] cluster_centers_8192.npy + 抽取后的小份锚点分数(不碰47G原始文件) ==="
    mkdir -p data/navsim/processed_data/anchor
    curl -L --fail -o data/navsim/processed_data/anchor/cluster_centers_8192.npy \
        "https://huggingface.co/liyingyan/DriveVLA-W0/resolve/main/cluster_centers_8192.npy"

    curl -L --fail -o pretrained_models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl \
        "https://huggingface.co/liyingyan/DriveVLA-W0/resolve/main/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
    curl -L --fail -o data/navsim/processed_data/train_vp_codes.zip \
        "https://huggingface.co/liyingyan/DriveVLA-W0/resolve/main/train_vp_codes.zip"
    (cd data/navsim/processed_data && unzip -q -o train_vp_codes.zip && rm train_vp_codes.zip)

    echo "--- 下载并抽取 47GB 原始锚点分数文件（只在这一步短暂落盘，抽取完可以删掉）---"
    curl -L --fail -o data/navsim/processed_data/anchor/formatted_pdm_score_8192.npy.zip \
        "https://huggingface.co/liyingyan/DriveVLA-W0/resolve/main/formatted_pdm_score_8192.zip"
    (cd data/navsim/processed_data/anchor && unzip -q -o formatted_pdm_score_8192.npy.zip && rm formatted_pdm_score_8192.npy.zip)

    # 内存充裕（HPC 一般 >64GB）时可以直接跑，不需要本机那套内存监控保护脚本；
    # 想稳妥一点还是可以用 tools/rl_prep/run_with_mem_guard.sh 包一层。
    python tools/rl_prep/extract_relevant_anchor_scores.py
    echo "抽取完成后可以删掉 47GB 原始文件，只留 trainval_scene_anchor_scores.pkl："
    echo "  rm data/navsim/processed_data/anchor/formatted_pdm_score_8192.npy"
}

download_rl_navsim_scenes() {
    echo "=== [rl_navsim_scenes] trainval navsim_logs（真实 NAVSIM PDM 仿真器奖励要用，见
    tools/rl_prep/navsim_score_worker.py）。假定 navsim_eval（地图）和 rl_anchor（trainval
    pkl + VQ codes）已经跑过 ==="
    mkdir -p navsim_data/navsim_logs
    (cd navsim_data && \
        curl -L --fail -o openscene_metadata_trainval.tgz "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz" && \
        tar -xzf openscene_metadata_trainval.tgz && \
        rm -rf navsim_logs/trainval && mkdir -p navsim_logs && mv openscene-v1.1/meta_datas navsim_logs/trainval && \
        rm -rf openscene-v1.1 openscene_metadata_trainval.tgz)
    echo "navsim_logs/trainval 装好了。接下来（在 navsim conda 环境下，见
    scripts/hpc/setup_navsim_env.sh）依次跑：
      1. (drivevla env) python tools/rl_prep/sample_trainval_scene_pool.py
      2. (navsim env)   python tools/rl_prep/cache_trainval_rl_scenes.py
    或者直接提交 scripts/hpc/submit_cache_trainval_rl_scenes.slurm（CPU 分区，不占 GPU）。"
}

download_nuplan_wm() {
    echo "=== [nuplan_wm] NuPlan world model checkpoint (~17G，阶段0.5用) ==="
    mkdir -p pretrained_models/DriveVLA-W0/Emu3_NuPlan_Pretrain_Cktps
    cd pretrained_models/DriveVLA-W0/Emu3_NuPlan_Pretrain_Cktps
    for f in config.json emu3.tiktoken emu3_vision_tokens.txt generation_config.json \
             model.safetensors.index.json special_tokens_map.json tokenizer_config.json \
             model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
             model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
        curl -L --fail -o "$f" "https://huggingface.co/liyingyan/DriveVLA-W0/resolve/main/Emu3_NuPlan_Pretrain_Cktps/$f"
    done
    cd -
}

case "$MODE" in
    core) download_core ;;
    navsim_eval) download_core; download_navsim_eval ;;
    rl_anchor) download_core; download_rl_anchor ;;
    rl_navsim_scenes) download_core; download_navsim_eval; download_rl_anchor; download_rl_navsim_scenes ;;
    nuplan_wm) download_core; download_nuplan_wm ;;
    all) download_core; download_navsim_eval; download_rl_anchor; download_rl_navsim_scenes; download_nuplan_wm ;;
    *) echo "未知模式: $MODE (可选: core / navsim_eval / rl_anchor / rl_navsim_scenes / nuplan_wm / all)"; exit 1 ;;
esac

echo "=== 完成 [$MODE] ==="
