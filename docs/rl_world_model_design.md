# 把 RL 接进 DriveVLA-W0：技术方案

> **当前进度（2026-10-09）**：见 [代码进度与实验结果](project_progress.md) 顶部的“全部结果一览”。结论摘要：原版RL在冻结navtest上PDMS −1.02 [−1.81,−0.30]，离线蒸馏与安全偏序均无可用增益，所有训练变体都是进度上升、TTC下降；测试时几何验证器选择（梯度提升）在留出log上 +2.07 [1.63,2.52] 且各成分不降，待在navtest上做端到端检验。以下保留历史设计，不代表仍在推进。

状态：草案 v0.7 · 阶段 0.5 + 阶段 0.5-bis(两种 world model 当 reward 的方式)均为负面结果,阶段 1/1.5(RL 机制、有界 KL、真实仿真器奖励)机制层面确认生效,阶段 2(训练前后 held-out 对比)显示单纯调 KL/reward 权重都堵不住"拿安全换 progress"这个 reward 走样(PDMS 0.9213→0.84~0.89 反复),阶段 3 改用 Safety-weighted GRPO(安全约束做硬门槛,不可被 progress 补偿)正在验证,见下方各节结果
适用代码库：本地 `DriveVLA-W0` 仓库(commit 见 `git log -1`)

## 阶段 1 结果(2026-08-01)：RL 机制本身在本机跑通了

代码见 [tools/rl_prep/rl_stage1_prototype.py](../tools/rl_prep/rl_stage1_prototype.py)。目的不是训出效果,是验证第 2-4 节说的这套链路——噪声注入让 flow matching 变成可求 log-prob 的随机策略、GRPO 组内相对优势、对冻结 reference policy 的 KL 约束、只训练解耦出来的 action expert——在代码层面是不是真的能跑通、梯度是不是真的在流动。

**结论:跑通了。** reward 暂时不用阶段 0.5 那个没验证过的 world model 信号,而是用仓库自带的、真实仿真器算出来的 8192 条锚点轨迹分数做"最近邻查表"——policy 采样出一条连续轨迹后,找 8192 条里最接近的一条,直接拿它的真实 PDM 分数当 reward,这样能把"RL 实现对不对"和"reward 好不好"这两件事分开验证。在 group size 4、4 个不同场景上跑了 4 步梯度更新,全部成功,没有崩:

```
step 000 reward_mean=9.15  reward_std=0.07  pg_loss=-3.71  grad_norm=764
step 001 reward_mean=3.31  reward_std=1.91  pg_loss=4.31   grad_norm=676
step 002 reward_mean=10.54 reward_std=0.04  pg_loss=-2.32  grad_norm=468
step 003 reward_mean=10.02 reward_std=0.03  pg_loss=6.11   grad_norm=564
```

`reward_std` 非零(同一组内采样出的轨迹质量确实不同)、`pg_loss`/`grad_norm` 都是有意义的非零值,说明采样→reward→组内优势→log-prob 重算→反向传播→参数更新这条链路是通的。

**这台机器(单卡 24GB,还跟其他进程共享)的实际能力上限**,踩了不少坑才摸清楚,记下来供以后参考:
- 满血 Emu3Pi0(VLM + action expert)本身在 bf16 下就要 ~18GB,这张卡装不下第二份完整模型——所以 reference policy 不能像最初设计的那样单独再 load 一份,得利用"VLM 本来就冻结不变"这个事实,只复制 581M 参数的 action expert 部分做 reference 快照,VLM 层两边共用。
- 对多步去噪做 log-prob 重算时,如果把整个多层网络当一个 checkpoint 单元,反向传播时还是会一次性摊开所有层的中间激活——必须像模型自己的 forward 一样,**逐层**做 gradient checkpointing,粒度不对等于没做。
- AdamW 默认给每个可训练参数配两份 fp32 状态,581M 参数就是 4.6GB,这张卡的余量(满打满算也就 4GB 出头)完全扛不住,换成 SGD 才够;连 SGD 的 momentum buffer(约 1.16GB,首次 `step()` 后才分配)都会让第二步开始 OOM,最终用的是 `momentum=0`。
- 就算上面全修完,单次前向+反向的 batch 也只能到 4(group size),再大(试过 8)还是会 OOM——这台机器的余量大概就卡在"4 个样本、6 步去噪"这个量级,再往上加(更大 group、更多去噪步、把 KL 约束打开)就需要第 7 节说的"更强算力"阶段了。

**目前这版跑的是简化配置**:`RL_KL_COEF=0`(为了多省出 1.16GB + 一次前向,还没验证 KL 项本身的数值行为)、SGD 无动量(不追求收敛质量,只看链路通不通)。换更大显存的机器后,应该做的事情按优先级排序:换回 AdamW、把 KL 约束打开、group size 提到 8 甚至更大、把阶段 0.5 里没跑完的完整交错格式和更大样本补上、最后再把 reward 从"最近邻查表"换成第 7 节说的在线仿真器打分。

## 阶段 1.5(2026-08-02)：KL 修复 + 换成真实 NAVSIM 仿真器奖励

阶段 1 在本机验证通了链路之后,换到 HPC(96GB RTX PRO 6000 Blackwell)上用 AdamW + group size 16 + KL 打开(`RL_KL_COEF=0.05`)跑了 200 步,奖励源仍是阶段 1 的"锚点最近邻查表"。结果暴露了两个问题：

**问题 1 — KL 无界增长。** 原实现是把每一步去噪、每个动作维度的 log-ratio 直接求和(240 项左右),再对这个和做 `exp()`。这个求和本身没有按项数归一化,量级会随 denoise 步数 × 动作帧数 × 动作维度线性增长,训练到 163 步时已经涨到 ~200——不仅数值上逼近 `exp()` 溢出,而且这么大的 KL 惩罚基本等于强行把 policy 摁回 reference,训练后期 `reward_mean` 从前 5 步均值 ~8.5 降到后 5 步均值 ~6.1,是明显的 reward collapse。

**修复**：改用 k3 estimator(John Schulman,http://joschu.net/blog/kl-approx.html)`(r-1) - log(r)`,恒 ≥0、方差更低；且改成对**逐元素**(除以 `steps × action_frames × action_dim`)的平均 log-ratio 算,量级稳定在 O(1),再 clamp 到 `[-20, 20]` 后才 `exp()`,不会溢出。`RL_KL_COEF` 相应从 0.05 调到 0.5(新旧两个系数作用在完全不同量级的量上,不可比)。代码见 [tools/rl_prep/rl_stage1_prototype.py](../tools/rl_prep/rl_stage1_prototype.py) 里 `recompute_log_prob` 调用处的 k3 计算。

**问题 2 — reward 信号本身仍是最近邻近似。** 阶段 1 的"snap 到 8192 个固定锚点之一,拿锚点的真实 PDM 分数当 reward"是为了先验证链路而做的简化,continuous 采样动作和最近锚点之间的偏差没有反映在 reward 里，会限制训练信号的精度。换成**在线真实 NAVSIM PDM 仿真器**打分：新增 [tools/rl_prep/navsim_score_worker.py](../tools/rl_prep/navsim_score_worker.py),跑在独立的 `navsim` conda 环境(numpy==1.23.4 / torch==2.0.1 / python 3.9,和 `drivevla` 环境的 torch 2.11 + numpy 2.x 不兼容)里,由 `rl_stage1_prototype.py` 通过子进程 + 文件 I/O 调用,一次打分一整个 GRPO 组。`RL_REWARD_SOURCE=navsim`(默认仍是 `anchor`,向后兼容)。

调试过程中踩了两个坑，都已修复：
- `MetricCacheLoader` 要传 `Path` 对象不能传 `str`。
- 最容易踩的一个：调用 `pdm_score(metric_cache, model_trajectory, future_sampling, simulator, scorer)` 时,第三个参数 `future_sampling` **必须是仿真器自己的 `PROPOSAL_SAMPLING`(40 点、0.1s 间隔)**,不是候选轨迹本身的采样规格(8/9 点、0.5s 间隔,即 `ACTION_SAMPLING`)。传错会导致内部把轨迹数组截断成只有 9 个点,但仿真循环仍按 40 步跑,索引越界后在 `batch_lqr.py` 的曲率 profile 计算里报 `ValueError: could not broadcast input array from shape (2,0) into shape (2,9)`。参考官方 `run_pdm_score.py:87` 的调用方式 `future_sampling=simulator.proposal_sampling` 才发现。用同一个已验证的 test-split metric_cache 交叉测试确认了这不是场景数据的问题,而是调用参数传错。

另外发现 DriveVLA-W0 自己的帧采样网格和 NAVSIM 官方场景网格没有严格对齐——本地从 300 个候选 trainval token 里过采样,只有 18 个真正落在合法的 NAVSIM 场景锚点上(用 `has_route=True/False` 测过,不是路由过滤导致的)。因此 `usable_indices` 不能再直接用 `anchor_scores.keys()`,改成用 [tools/rl_prep/cache_trainval_rl_scenes.py](../tools/rl_prep/cache_trainval_rl_scenes.py) 实际筛出并建好 metric_cache 的 `trainval_rl_scene_tokens_cached.json` 场景池。

**本机验证**：`navsim_score_worker.py` 独立测试(3 条合成轨迹)返回了真实、有区分度的 PDM 分数。完整脚本(`RL_REWARD_SOURCE=navsim`)跑通了模型加载、18 场景池加载、rollout、navsim 打分子进程调用(均无报错)；随后在 `recompute_log_prob` 的反向传播里 OOM——这是本机 24GB 卡的显存余量问题(仅模型权重就占 19GB,当时桌面上还跑着 CARLA 仿真器等其他进程占用 ~2.6GB),不是新代码的 bug,和阶段 1 记录的"这台机器的实际能力上限"是同一类已知限制。

**HPC 部署**：新增 [scripts/hpc/setup_navsim_env.sh](../scripts/hpc/setup_navsim_env.sh)(建 `navsim` conda 环境)、`fetch_data.sh` 的 `rl_navsim_scenes` 模式(下载 trainval navsim_logs)、[tools/rl_prep/sample_trainval_scene_pool.py](../tools/rl_prep/sample_trainval_scene_pool.py)(在 `drivevla` 环境里从 dataset 采样候选 token 池,替代本机当初手写的一次性脚本)、[scripts/hpc/submit_cache_trainval_rl_scenes.slurm](../scripts/hpc/submit_cache_trainval_rl_scenes.slurm)(CPU 分区一次性任务,串联上面两步生成 HPC 自己的场景池 + metric_cache)。`submit_rl_stage1.slurm` 已更新为 `RL_REWARD_SOURCE=navsim` + `RL_KL_COEF=0.5`。

**HPC 部署过程中另外踩的坑**（都已修复，写进对应脚本了）：trainval navsim_logs 的 tgz 解压出来比 test split 多套一层目录（`navsim_logs/trainval/trainval/`，手动改成单层）；`standard` 分区的计算节点会把 module 系统自带的 python 排到 conda env 前面，导致 `import tiktoken` 失败（`fetch_data.sh` 里本来就有的 `export PATH="$CONDA_PREFIX/bin:$PATH"` 保险，两个新 sbatch 脚本一开始漏加了，已经补上）；`$(conda info --base)` 在 module load 之后解析到的是 HPC 系统自带的 conda 装载路径而不是用户自己的 `~/.conda`，导致 `NAVSIM_ENV_PYTHON` 拼错路径，改成直接写死 `$HOME/.conda/envs/navsim/bin/python`。

**HPC 跑通结果（2026-08-03，job 55983182，28 个真实 trainval 场景，200 步，group size 16，AdamW）**：

```
=== Summary ===
reward_mean over training: first 5 = [0.9134, 1.0000, 1.0000, 0.9662, 1.0000]
                             last 5 = [1.0000, 1.0000, 0.8270, 0.9767, 0.9696]
```

**KL 全程有界**：从 0 起步，在 200 步里稳定在 0.0006~0.034 之间波动，没有再出现旧版本 163 步涨到 ~200 的无界增长——这次改动最直接要验证的东西，确认生效。

**没有再出现 reward collapse**：首尾 5 步均值 0.976 → 0.955，基本持平，不是旧版本那种 8.5→6.1 的明显下滑趋势。不过这个对比本身意义有限，因为不同场景难度差异很大（比如 `a5efa651fec451b5` 常年在 0.2~0.6 波动，多数场景直接拿满分 1.0），首尾 5 步抽到哪些场景，对均值的影响比训练效果本身更大。

**这轮结果的局限，不代表"RL 已经证明能提升驾驶表现"**：只有 28 个场景（300 个候选里只有这些落在合法的 NAVSIM 场景锚点上），200 步里被反复抽样，本质上还是阶段 1 那种小规模场景过拟合测试，不是有统计意义的性能提升实验；没有做训练前后的独立 held-out 评测；基线模型（87.2 checkpoint）本身很强，多数场景奖励已经接近满分，这个设置下天花板本来就低。

**下一步（阶段 2）**：把候选场景池从 300 扩大（调大 `sample_trainval_scene_pool.py` 里的 `RL_SCENE_POOL_SIZE`，现在只筛出 28/300 有点少）、留出一部分场景做训练前后的独立对比评测（复用 `run_pdm_score.py` 那套流程），而不是现在这种训练和评测用同一批场景的设置，才能真正回答"RL 有没有提升驾驶表现"这个问题。

## 阶段 3(2026-08-05)：从"调权重"转向"约束优化"——Safety-weighted GRPO

阶段 2 收尾时试过两次单纯改数字的修法,都没解决根本问题:
- 收紧 KL(0.5→1.5,3倍):退化在 100 步内就定型,不随训练时长恶化——问题不在 KL 松紧。
- 把训练 reward 的 TTC:progress 从官方 5:5 改成 8:2 加权:只是换了"安全换进度"的兑换比率,没有堵住这个漏洞本身,progress 涨得够多依然能把安全分的损失换回来。

**改成约束优化的思路**(不再用单一加权标量当 reward):
- performance reward:ego_progress + driving_direction_compliance
- safety constraints(硬门槛,不是可以拿 progress 补偿的软惩罚项):no_at_fault_collisions、drivable_area_compliance、time_to_collision_within_bound
- regularization:对 reference policy 的 KL(已有)

第一版实现 Safety-weighted GRPO(`compute_safety_gated_advantage`,`tools/rl_prep/rl_stage1_prototype.py`):把候选组按是否满足安全门槛(NC=1.0 且 DAC=1.0 且 TTC≥0.5)分成 safe/unsafe 两组——unsafe 候选无论 progress 多高,advantage 恒为负,永远拿不到正向梯度;safe 候选内部才做标准 GRPO 组内相对排序(按 progress + driving_direction 排序)。全员 unsafe 的退化情况(全组没有一个安全候选)用违规程度的组内相对排序兜底,保证还有梯度信号,不会整步梯度为零。单测覆盖了 4 种边界情况(混合安全/不安全、全不安全、只有 1 个安全候选、全安全退化到标准 GRPO),见 session 记录。

`navsim_score_worker.py` 保留了阶段 2 那版 8:2 加权的 `train_reward` 字段(现在只当对照,不再被训练消费),新增了 `driving_direction_compliance` 字段供 performance reward 使用。

规划中的后续(按依赖顺序,还没做):
1. Constrained/Primal-Dual Flow-GRPO(主方法):把硬门槛换成拉格朗日对偶变量软约束(`λ_c`、`λ_t` 动态调整),预算不足以支撑硬门槛时也有梯度信号,而不是非黑即白。
2. Risk-aware 场景采样(高 TTC 风险/路口/困难场景过采样),必须等 reward/advantage 设计验证完再加,不然分不清效果来自哪个改动;held-out 评测集(286 个,原始分布)保持不动,不做任何 risk-aware 调整,继续当所有版本共同的裁判标准。
3. 四版对比消融:标准 GRPO / KL-GRPO / Safety-weighted GRPO / Constrained Flow-GRPO。

## 阶段 2 结果(2026-08-04)：真正的训练前后 held-out 对比——PDMS 下降了

场景池扩容后(300→3000 训练候选、另建 4000 候选的 disjoint held-out eval 池),训练池筛出 228 个有效场景,eval 池筛出 286 个,互不重叠。GPU 分区(96GB)上跑了 1000 步(`RL_KL_COEF=0.5`,20 小时时限内实际跑到第 821 步被系统终止,最后一次存档在第 799 步)。同一批 286 个 held-out 场景,训练前后各跑一次完整 PDMS 评测(见 `scripts/hpc/submit_eval_subset.slurm`,复用 `run_pdm_score.py` 原生打分流程):

```
训练前(87.2 SFT checkpoint):  score=0.9213  NC=1.000 DAC=0.976 EP=0.852 TTC=0.993 comfort=1.000 DDC=0.993
训练后(第 799 步 checkpoint):  score=0.8591  NC=0.969 DAC=0.955 EP=0.894 TTC=0.850 comfort=0.986 DDC=0.991
```

**PDMS 从 0.9213 掉到 0.8591,下降约 6.2 个百分点,是负面结果。** 拆开看:`ego_progress` 是唯一变好的子项(0.852→0.894),`time_to_collision_within_bound` 跌得最狠(0.993→0.850),`no_at_fault_collisions`/`drivable_area_compliance`/`comfort` 都小幅变差。这个模式符合"reward 走样"的典型特征——NAVSIM 官方 PDMS 公式本身把 TTC 和 ego_progress 同权重(`5×TTC + 5×EP`),policy 学会了用"更激进地往前冲"换分,但没有真正学会处理近距离交互场景的安全裕度,在 TTC 这个维度上以拉低真实驾驶质量为代价拿到了更高的 reward。

**存档逻辑的一个 bug**:第 199/399/599/799 步分别存过 checkpoint,但当时四次存档用的是同一个固定路径,后面的存档直接覆盖了前面的——只有最后一次(第 799 步)的权重留了下来,没法追溯退化具体从哪一步开始。已修复(`rl_stage1_prototype.py`,每个 checkpoint 存到按步数命名的独立子目录)。

**下一步**:不重跑旧配方去单纯"看曲线"(成本太高,而且已知这个配方效果不好)。直接把 `RL_KL_COEF` 从 0.5 收紧到 1.5(限制 policy 偏离 SFT 参考模型的幅度,不改动 reward 公式本身,训练目标依然是纯官方 PDMS)、步数从 1000 缩到 400、`RL_SAVE_EVERY` 从 200 缩到 100,一次训练里就能同时验证"收紧 KL 是否缓解了这个问题"和"退化从哪一步开始"。这次的“world model amplifies RL”式 scaling law 研究**明确暂缓**——阶段 2 是"gate",只有 RL 出现正向提升才值得投入做多规模对比,当前是负结果,先诊断修复。

## 阶段 0.5-bis 结果(2026-08-03)：换一种 world-model-as-reward 机制,仍是负结果

阶段 0.5 测的是"world model 对候选轨迹的生成保真度"(teacher-forced NLL)。这里换一种更接近 VLA-RFT "verified reward" 机制的操作化方式:不问"这条轨迹像不像 world model 见过的东西",而是把 world model 当前向模拟器——分别用候选动作和专家(GT)动作条件化模型,取 EOA token 位置的隐藏状态(不需要真的解码出未来帧,单次前向就够,比自回归生成候选轨迹的未来视频帧便宜得多),用两者的 cosine/L2 距离衡量"想象出的未来离专家的未来有多远"。代码见 `tools/rl_prep/stage0_5b_rollout_distance.py`,同样在 8 场景 × 40 候选(320 条,和阶段 0.5 同规模、可直接比较)上测。

**结论:同样没有可靠信号。**

```
vs combined PDM-like score: spearman=-0.038 (p=0.501)  pearson=-0.057 (p=0.309)
```

方向对了(距离越大分数应该越低,这次是负相关),但强度可以忽略,且不显著——和阶段 0.5 的 spearman=0.051 是同一量级的空结果。子指标里 `drivable_area_compliance` 是唯一方向对且显著的(spearman=-0.148, p=0.008),但 `no_at_fault_collisions`/`time_to_collision_within_bound` 反而显著地方向错(距离越大反而"越安全"),说不通,更像噪声。混杂检验还发现这个隐藏状态距离本身和"候选动作与专家动作在数值空间里的 L2 距离"显著相关(spearman=0.183, p=0.001)——说明这个信号很大程度上只是在间接反映动作数值差异,不是真的捕捉到了想象中的驾驶结果好坏。

**两次独立验证(生成保真度、前向推演距离)都指向同一个结论**:DriveVLA-W0 这个 world model(纯 next-frame-prediction 目标训出来的,没有做过 goal-conditioning 或 reward 相关训练)不具备支撑"直接当 RL reward"这类判别任务的能力,不建议再尝试第三种操作化方式。真实仿真器(阶段 1.5)仍是唯一验证有效的 reward 来源。

## 阶段 0.5 结果(2026-08-01)

用 `tools/rl_prep/stage0_5_worldmodel_fidelity.py` 在 8 个 trainval 场景 × 40 条分层采样的候选轨迹(共 320 条)上,测了"world model 对真实未来帧的 teacher-forced NLL"和"该候选轨迹的真实 PDM 子分数"之间的相关性。**结论:没有可靠信号,建议在投入第 2-4 节的 RL 实现之前重新评估第 3.1 节的 reward 设计。**

**混合在一起看(320 条,不分场景)**:combined PDM-like score 和 NLL 的 pearson 相关是 -0.164(p=0.003,方向对——NLL 越高、PDM 分数越低),但 spearman 只有 0.051 且不显著,说明这个弱相关可能是被少数极端值带动的,不是稳定的单调关系。分指标看更乱:`drivable_area_compliance` 方向对且弱显著,`comfort` 却显著地**反方向**(NLL 越高、舒适度分数反而越高),`no_at_fault_collisions`/`ego_progress`/`time_to_collision` 都不显著。

**按场景内(GRPO 真正用得上的口径)看,信号基本消失**:GRPO 算的是同一场景内候选之间的相对优势,所以更该看的是"场景内 NLL 排序和场景内 PDM 排序是否一致"。8 个场景各自算 spearman 后,`combined score` 的场景内 rho 是 `[0.32, -0.37, -0.36, -0.47, 0.47, -0.09, 0.21, -0.26]`——正负都有,均值 -0.066、中位数 -0.175,本质上是围绕零的噪声,没有一致方向。

**混杂因素排除得还算干净**:NLL 和"候选轨迹到专家轨迹的 L2 距离"没有显著相关(spearman=-0.033, p=0.556),说明 NLL 不是在简单重复"像不像人类轨迹"这个已经被距离项吃掉的信号,是相对独立的量——但独立不代表有用,这次测出来它对真实 PDM 分数的解释力也很弱。

**需要区分两个不同的问题,原方案在这里有概念混淆**:论文附录里"生成保真度与规划表现正相关"的发现,大概率是**模型级别**的观察(训得更好的 checkpoint,生成质量和规划质量同步变好),而阶段 0.5 测的是**同一个 checkpoint 内、同一场景下、候选轨迹级别**的判别力——这是两个不同粒度的问题,前者成立不代表后者也成立。这次的结果更像是在说后者不成立,或者至少用这种探测方法测不出来。

**方法论上可能站不住的地方**(在得出"这条路走不通"的结论前,值得先排除):
1. 只用了单帧 teacher-forced NLL(4 秒后那一帧),丢弃了中间帧和历史动作条件,可能抹掉了大部分可判别的信息——第 6 节里提到的完整 6-pair 交错格式也许信号更强,但计算成本高很多。
2. 8 个场景对"场景内相关性"这个统计量来说样本太小,场景内 8 个 rho 值本身方差就很大,不能排除是采样噪声掩盖了真实但更弱的信号,需要更大样本才能下确定性结论。
3. 候选轨迹是全局固定的 8192 条聚类中心,不是针对每个场景专门生成的候选,场景内的动作空间覆盖可能不够均匀。

**建议的下一步**(不是自动执行,需要你决定优先级):要么扩大样本(更多场景/更多候选)重新测,要么换成第 6 节说的完整交错格式测一次,要么先接受"这个具体的保真度信号不够用",转向第 5 节里几何安全代理为主、world model 只做很小权重的辅助信号这个更保守的设计。

代码见 [tools/rl_prep/stage0_5_worldmodel_fidelity.py](../tools/rl_prep/stage0_5_worldmodel_fidelity.py),原始数据见 `docs/stage0_5_results.json`。

## 0. 先说结论

核心主张不变:**复用 DriveVLA-W0 自己训出来的 world model,作为 RL 阶段的内部模拟器/reward 来源,零额外训练成本**。但读完这份仓库实际的模型代码后,有两个会直接决定架构走向的事实,原提案里没有覆盖,必须先讲清楚:

1. **我们本地能跑的 87.2 分 flow-matching checkpoint,不是 world model。** 它的 forward 里 `vlm_loss_weight` 被硬编码成 `0.0`([modeling_emu3.py:2129](../reference/Emu3/emu3/mllm/modeling_emu3.py#L2129)),也就是说这个 checkpoint 训练/推理时压根没有做未来帧预测,只做 flow matching 动作回归。真正做"预测未来帧"的 world model,是 Stage 1 NuPlan 预训练阶段的自回归/MoE 路径(`Emu3MoE`/`Emu3AutoRegressive`,对应 `Emu3_NuPlan_Pretrain_Cktps` 这个 checkpoint,以及 `train_nuplan_6va_multi_v0.2_worker.sh` 里训的模型)。
2. **仓库里已经有一套"用 PDM 子指标做 reward"的离线机制,但不是在线 RL。** `Emu3QFormer._compute_anchor_loss`([modeling_emu3.py:2960](../reference/Emu3/emu3/mllm/modeling_emu3.py#L2960))对应 README 里的 `Emu3_Multi_Anchor_PDMS_90.3` / query-based 变体:用一个固定的 8192 条轨迹词表(`cluster_centers_8192.npy`)+ 每条轨迹预先用 NAVSIM 仿真器算好的 PDM 子分数(`formatted_pdm_score_8192.npy`,NC/DAC/EP/TTC/Comfort),构造一个 reward-加权的 soft target 分布,做蒸馏式交叉熵训练——这本质是 Hydra-MDP 的做法:reward 来自**外部仿真器离线预计算**,不是 world model,也不是在线 rollout。`Emu3AutoRegressive.generate_actions` 里还留了个 `grpo_sample=True` 的采样开关([modeling_emu3.py:3705](../reference/Emu3/emu3/mllm/modeling_emu3.py#L3705)),但训练循环本身不在这个"reviewed 版本"仓库里,无法确认是否真的接了在线 GRPO。

结论:你提的"world model 当 RL reward"这个缺口**确实存在**,但要把它做对,得先在"用哪个 checkpoint 当 world model""怎么把它接到 RL 的 reward 函数里"这两件事上做出明确选择——这是本方案的主体。

## 1. 角色划分:谁是 policy,谁是 world model

| 角色 | 用哪个 checkpoint/模型类 | 代码位置 |
|---|---|---|
| **Policy(RL 微调对象)** | `Emu3Pi0` flow-matching Action Expert,从 `Emu3_Flow_Matching_Action_Expert_PDMS_87.2` 初始化 | [modeling_emu3.py:1926](../reference/Emu3/emu3/mllm/modeling_emu3.py#L1926) |
| **Reference policy(KL 锚点)** | 同一个 87.2 checkpoint 的冻结拷贝 | 同上 |
| **World model(reward 来源)** | `Emu3MoE`/`Emu3AutoRegressive`,从 `Emu3_NuPlan_Pretrain_Cktps` 初始化,只做未来帧 token 的似然/重建评估,不参与梯度更新 | [modeling_emu3.py:1439](../reference/Emu3/emu3/mllm/modeling_emu3.py#L1439)、[modeling_emu3.py:3268](../reference/Emu3/emu3/mllm/modeling_emu3.py#L3268) |
| **几何安全代理(防 reward hack)** | 轻量、非学习的碰撞/越界检测,直接用 navsim 的地图 API 和场景里的他车 box(不依赖仿真器闭环) | 复用 `inference/navsim/navsim` 里的地图/box 数据结构,自己写打分函数,不跑完整 PDM 仿真 |

选 Action Expert 而不是 AR/anchor 头当 policy,是因为它是 500M 的解耦小模块(README 里明确说这是从主干解耦出来的),天然适合冻结 VLM 主干只对它做 RL,训练成本最低,这点和原提案第 4 点一致。

## 2. 把 flow matching 变成可以做策略梯度的随机策略

`Emu3Pi0.sample_actions`([modeling_emu3.py:2207](../reference/Emu3/emu3/mllm/modeling_emu3.py#L2207))现在是**确定性** Euler ODE 积分:从 `z~N(0,I)` 出发,`dt=-1/N`,每步 `z = z + dt * v_θ(z,t)`,没有随机性,也就没有一个可以求导的 log-prob——没法直接套策略梯度。

底层的 rectified flow 训练目标在 [flow_matching.py](../models/policy_head/flow_matching.py):`zt = (1-t)*x + t*z1`,目标速度 `z1 - x`,标准 conditional flow matching,和 ReinFlow 论文处理的对象完全一致,可以直接照搬它的做法:

- 把确定性 Euler 步改写成 `z_{t+dt} = z_t + dt * v_θ(z_t, t) + σ(t) * ε`,`ε~N(0,I)`,即在每一步去噪里注入一个小的、随时间步衰减的高斯噪声。
- 这样每一步就变成一个高斯转移核,`log π(z_{t+dt} | z_t)` 可以解析算出来(ReinFlow 论文式 3-5 的做法),整条采样轨迹的 log-prob 就是各步 log-prob 求和。
- 采样步数(`action_sample_steps`,当前 checkpoint 用的是多少步需要在 config.json 里确认)决定了这条"轨迹"的长度——步数越多,方差累积和梯度计算开销越大,是一个要调的超参。

这一步是纯工程实现,风险可控,ReinFlow 已经把数学部分证明过了,不算创新点,但必须先做,否则后面 GRPO 无从谈起。

## 3. Reward 设计:分层,不要只信 world model

### 3.1 主信号:world model 预测保真度

对 policy 采样出的每条候选动作:
1. 把动作转换成 ego 未来位姿(仓库里 `EmuVLAAgent.convert_relative_to_absolute_se2_trajectory` 已经有这个逻辑,[emu3_agent.py](../inference/navsim/navsim/navsim/agents/emu_vla_agent.py#L52)可以直接抄)。
2. 用 world model(`Emu3AutoRegressive`/`Emu3MoE`)以"当前帧 + 该候选动作隐含的未来自车状态"为条件,自回归地打出下一帧 VQ token 序列的对数似然,或者直接用 `forward_action` 同款结构算重建误差。
3. 用这个似然/误差的**负值**当 reward 的一项——对应论文附录"生成保真度与规划表现正相关"的发现,这是全文里唯一一个和 DriveVLA-W0 自己训练产物强绑定、且零额外训练成本的信号。

### 3.2 辅助信号:几何安全代理(必须加,理由见第 5 节)

不依赖 world model,直接用场景里已知的地图可行驶区域和他车真实轨迹(navsim 场景数据里就有),对候选轨迹做:
- 是否驶出可行驶区域(对应 DAC)
- 是否与他车 box 在未来几步内重叠或跌破安全距离(对应 NC/TTC 的简化版)

这两个可以用纯几何计算,不需要跑 `run_pdm_score.py` 那套完整仿真,计算成本很低,可以放进 reward 里当乘法项或惩罚项(仿照 PDMS 公式里 NC/DAC 是乘子这个设计,而不是简单加权求和)。

### 3.3 合成

```
reward = (geometry_safety_multiplier) * (w1 * world_model_fidelity + w2 * comfort_penalty)
```

`geometry_safety_multiplier` 取 0/0.5/1 三档(仿照真实 PDMS 里 NC/DAC 的离散乘子设计),严重违规直接把 reward 打成 0,而不是让 world model 的高保真度分数把违规"淹没"掉——这是防 reward hack 的第一道闸。

## 4. RL 算法:组内相对优势(GRPO 风格),不单独训 critic

对每个场景采样 G 条候选动作(复用第 2 节的随机化 flow 采样),各自算 reward,组内做

```
A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)
```

policy loss 用 `-A_i * log π_θ(traj_i)`(log-prob 是第 2 节各去噪步 log-prob 之和),这样不需要单独训一个 value/critic 网络,和 ReCogDrive 的 DiffGRPO、VLA-RFT 的 GRPO 用法一致,是这四篇论文里最成熟、风险最低的部分,直接抄配方即可。

在这之上加一个**对 reference policy(冻结的 87.2 checkpoint)的 KL 惩罚**:flow matching 没有离散 token 那种现成的 KL,用 ReinFlow 论文的近似——同一批噪声下 policy 和 reference 预测速度场的 MSE 差,作为 KL 的代理项,乘个系数加进 loss。这是防止 policy 被 world model reward 带偏、跑到 world model 也没见过的分布之外的第二道闸。

## 5. 风险:World model 会不会被"骗"

这是最大的技术风险,原提案第 5 点已经点到,这里展开成可执行的检验计划:

**假设**:world model 的预测保真度和"动作是否安全合理"正相关,这个相关性大概率是在专家轨迹(训练/eval 集里的真实人类驾驶行为)上测出来的。RL 探索必然会把动作推向这个分布之外——正好是相关性最可能失效的区域,因为 world model 从没见过这些输入,它的似然/重建误差在 OOD 输入上本身就是不可靠的(可能"自信地生成错误预测",也可能"到处都不自信")。

**在写任何 RL 代码之前,先回答这个问题**——这是第 0 阶段,细节见第 6 节。

**如果验证后相关性在 OOD 区域确实弱或失效**,缓解手段(按实现成本排序):
1. 加大第 4 节的 KL 惩罚系数,把探索半径锁死在 world model 还算可靠的邻域内。
2. World model 换成一个轻量 ensemble(比如两个不同 dropout mask 或不同 checkpoint 的预测),预测方差大的样本直接丢弃或 reward 打折——不信任 world model 自己都不确定的判断。
3. 第 3.2 节的几何安全代理权重调高,退化成"world model 只管舒适度/合理性这类软指标,硬安全约束完全交给不会被骗的几何检测"。

## 6. 分阶段验证计划

不要一上来就搭完整 RL loop,按下面顺序推进,每一步都是前一步的必要条件:

**阶段 0 — 离线相关性验证(1天内,不用新代码)**
用已经跑出来的 `output/PDMS_87.2/` 结果(12146 条 JSON)+ world model checkpoint:对每条测试样本,用 world model 算一次"预测保真度"分数,和该样本对应的(需要额外跑一次)PDMS 分数做相关性分析。这一步直接回答"这条路走不走得通",而且完全复用现有产物,不需要写训练代码。

**阶段 0.5 — 白嫖仓库里已有的 anchor 数据做同一件事**
`cluster_centers_8192.npy` + `formatted_pdm_score_8192.npy` 本身就是"(候选轨迹, 真实 PDM 子分数)"的现成数据集,而且是仿真器给的、比阶段 0 更权威的 ground truth。可以直接用这批数据,对每条 anchor 轨迹跑一次 world model 保真度评分,看它和仿真器给的真实 PDM 分数相关性如何——这是对第 3.1 节假设最直接、最省事的检验,建议和阶段 0 一起做。

**阶段 1 — 单场景过拟合测试**
固定一个场景,跑通"采样 G 条动作 → world model 打分 → 算 advantage → 更新 policy"全链路,只看 loss 能不能下降、policy 会不会坍缩,不追求真实效果。目的是让 RL loop 本身的实现(第 2、4 节)先跑通,和 reward 设计的好坏解耦开调试。

**阶段 2 — 小规模在线 RL**
在 navtest 的一个子集(比如 100-200 个场景)上跑真实的在线 RL,和 SFT baseline 对比 PDMS。这一步才需要认真考虑第 7 节的算力问题。

## 7. 算力现实

本机只有 1 块 GPU(RTX PRO 4000 Blackwell,24GB)。作为参照:刚才用这块卡跑 87.2 checkpoint 在全量 navtest(12146 条)上的**确定性单次推理**,耗时约 9 小时 10 分钟。

RL 训练每个 step 需要:采样 G 条动作(G 次 flow ODE 积分,每次仍是多步去噪)→ world model 对每条候选都做一次未来帧似然/重建评估(如果 world model 是自回归的,这一步单条样本的开销可能不比 flow 采样便宜)→ 反向传播更新 policy。同一个"场景"上的计算量级,粗略估计是确定性推理的 `G × (world model 前向次数)` 倍。

在真正写 RL 训练循环之前,建议先把"G 取多少、world model 用几步生成、batch size 多大,在这块卡上一个 step 实际要多久"这几个数字测出来(可以用阶段 1 的单场景实验顺带测),再决定阶段 2 用多大规模、要跑多久,避免方案在算力上直接卡死。

## 8. 暂不建议做的部分

- **原提案第 3 点(RL 收益是否也随数据规模反转)**:需要在多个数据规模上重新做 SFT+RL 对比,是独立的大实验,和第 1/2 点的核心贡献不在一个量级,放 future work,不要现在就规划实现。
- **"70M 帧"这个数字**:仓库 README 里没有对应说明(NAVSIM 训练集写的是约 10 万帧,NuPlan 预训练规模未注明),写进正式材料前建议对着论文原文核实这个数字具体指哪个训练阶段的数据量。

## 9. 需要你决策的开放问题

1. World model 打分用**似然**(需要 AR 逐 token 算概率,准但慢)还是**重建误差**(用 `forward_action` 同款结构算 MSE,快但对"合理性"的刻画更粗糙)?建议阶段 0 两个都测,用实测相关性强弱来选,而不是先验决定。
2. 第 3.3 节的几何安全代理,是自己写一个轻量版(纯 Python 判几何重叠/可行驶区域),还是复用已经装好的 navsim/nuplan-devkit 里现成的碰撞检测组件(依赖变重,但少踩重复造轮子的坑)?
3. 阶段 0 的 PDMS 分数,是复用之前跑通的 `run_pdm_score.py`(准确,但每次验证都要过一遍完整仿真流程),还是先用几个简单的启发式代理(比如轨迹终点是否偏离 GT 太远)做快速筛选,留到确认方向后再上真仿真?
