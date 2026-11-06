# 原版五个检查点：冻结 navtest 回归评估

状态：已在 HPC 运行完成（作业 63561267–63561274，23 个臂全部成功，SFT 臂复现 88.749）。只推理和评分，不训练、不改系数、不重新选点。

**结果（1000 场景/136 log）**：五个原版检查点平均 PDMS −1.023，按 log 的 95% 区间 [−1.813, −0.300]，判定 `GAIN_DOES_NOT_EXTEND`；NC −0.58、DAC −0.68、TTC −1.80、Progress +0.53；五个种子全为负；第 10/25/40/50 步分别 −0.71/−0.78/−0.88/−0.66。详见 `docs/iclr2025_conference.tex` 的 navtest 小节。

## 目的
原版五种子结果（直接 RL，无投影）在 286 场景上 PDMS +0.604 [−0.21, +1.45]、Progress +2.01、TTC −0.70。本包把那五个按 150 场景验证选定的检查点放到之前评测过后来 RL 的同一份冻结 1000 场景 navtest（136 logs）上，判断原版的“进度涨、TTC 降”是否在更大集合上也出现。

## 与 `rl_il_navtest_v1` 相同的部分
冻结场景/映射/协议哈希、cache 与原始 log 只读复用、`eval.sh` 评测代码、评测种子 0、repo 快照（`rl_pareto_20261001_182743_81761/repo`）。先 CPU 审计，再 SFT 8 场景 smoke，通过后才评测。

## 不同的部分
- 检查点：`direct_rl_seed20–24` 的第 25/25/50/10/40 步（主分析）；其余 15 个保存步数只作描述；SFT 另用评测种子 1、2 重复评测以量出评测噪声。共 23 个臂（`arms.py`）。
- 隔离检查：原版训练池是全部缓存的 trainval 场景（`metric_cache_trainval_rl`），所以检查 navtest 的 log 与整个训练池无重叠（已在 HPC 手动核实为 0/136，`prepare.py` 再核一次）。
- 这些检查点没有 `training_verified.json`，训练预算不在此处核对；配置来源见论文表 `tab:protocol`。
- 报告 `report.py`：五个检查点平均对 SFT 的 PDMS 差 + 整 log 配对 bootstrap；事先固定判定：区间下界 >0 为 `EFFECTIVE_ON_NAVTEST`，上界 <0 为 `GAIN_DOES_NOT_EXTEND`，其余 `INCONCLUSIVE`。

## 运行
```bash
bash run.sh submit     # Mac；需要 Okta 验证
bash run.sh fetch      # 作业完成后
```
提交前确认 `rl_pareto_20261001_182743_81761/repo` 与 `rl_pareto_navtest_v2_20261005_133703_37354` 仍在 HPC 上，且没有人在清理它们。首个 SFT 臂在 navtest 上应复现 88.749（之前的评测，确定性）；复现不了就不要解读后面的结果。

## 边界
navtest 子集已被反复查看，不是盲测；SFT 来源未核实；区间只覆盖场景/log 抽样，不含训练随机性；不能用本结果挑 checkpoint 或调参。
