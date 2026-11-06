# 复现清单:论文中每个数字从哪里来

范围缩写:**FULL** = 官方 navtest 全量(12146 场景、136 个日志);**SUB** = 1000 场景冻结子集(136 个日志);**HO600** = 600 个留出 trainval 场景。
目录均在 `/gpfs/accounts/xinxia_root/xinxia0/huamin/DriveVLA-W0-data/navsim_data/exp/`(下记 `$X`)。脚本在 `scripts/hpc/` 与 `tools/rl_prep/`。

## 0. 公共资源

| 资源 | 位置 / 来源 |
|---|---|
| 官方度量缓存(FULL) | `$X/metric_cache_navtest_official`,脚本 `submit_official_navtest_cache.slurm`,作业 64063343 |
| FULL 队列目录(含加了 token 的 pkl、token 列表、日志映射) | `$X/official_navtest_full_cohort2`,脚本 `prepare_official_cohort.py` |
| 官方评分 | `submit_official_navtest_score.slurm`(`run_pdm_score.py` + `emu_vla_agent`) |
| 官方推理 | `submit_official_navtest_infer.slurm`(`MODEL=flow|query|anchor`),补丁启动器 `run_inference_anchor_patch.py` |
| 候选池(训练选择器) | `$OFFLINE/navsim_data/exp/world_action_full_v1`(7171 场景 × 64 候选),几何 `world_action_geom_full_v1`,TTC 块 `world_action_ttc_full_v1` |

## 1. 官方基线复现(FULL)

| 数字 | 输出 | 作业 |
|---|---|---|
| 流匹配 SFT,我们的推理 86.75 | `$X/official_navtest_flow_sft/2026.10.15.17.53.29/*.csv` | 推理 63865113,评分 64084917 |
| 作者发布的输出 87.01 | `$X/authors_flow_87p2/2026.10.15.19.29.47/*.csv` | 评分 64094583 |
| Query-based 90.26 | `$X/official_navtest_query_sft` | 推理 64098368,评分 64357100 |
| Multi-Anchor 90.33 | `$X/official_navtest_anchor_sft/2026.10.20.15.04.26/*.csv` | 推理 64357328,评分 64367701 |
| 配对比较(我们 vs 作者,−0.26) | `$X/pair_ours_vs_authors_flow.json` | `analyze_paired_navsim_scores.py` |

注:作业号取自提交时的回显;以 `sacct -j <号>` 为准。

## 2. 选择器(FULL,流匹配基座)

| 数字 | 输出 | 作业 / 命令 |
|---|---|---|
| 选择器主结果(HGB +3.00 / +2.90 对独立 SFT) | `$X/navtest_selector_e2e_20261020_201323_full/selector_navtest.json` | 收集 64401276,几何 64401277,拟合 64401278(`run_navtest_selector_e2e.sh full`,`NT_ARRAY=0-47%8`) |
| 官方入口复核 89.65 | `$X/official_navtest_selected_hgb` | `export_selected_to_official_json.py` + 评分 64478535 |
| 配对检验(对规则、仅几何、仅轨迹、ridge) | 同上 JSON | `compare_within_selector.py` |
| 特征拆分(地图 / 他车) | `.../selector_navtest_split.json` | 64490168(`--split-geometry`) |
| 碰撞时间块 | `.../selector_navtest_ttc.json` | 提取 64504197,拟合 64504198(`run_ttc_block_navtest.sh`) |
| 地图噪声 0.2 / 0.5 / 1.0 m | `.../selector_mapnoise{1,2,3}.json` | 拟合 64504200 / 202 / 204(`run_map_noise_navtest.sh`) |
| TTC 块的配对检验 | — | `compare_selector_runs.py selector_navtest.json selector_navtest_ttc.json hgb` |

## 3. 训练类方法

| 数字 | 范围 | 来源 |
|---|---|---|
| 原始 RL(5 个检查点)等 | SUB | `docs/project_progress.md` 与此前账本;检查点在备份 `/nfs/turbo/umd-xinxia/huamin_backup/purge_20261006_job_64322695` 与 `posttrain_runs` |
| 偏好蒸馏(PDMS / 间隔 / 最近安全) | SUB | `submit_pair_pref.sh`,汇总 `pair_pref_comparison.json` |
| 离线蒸馏(无条件 / 条件化 / 过滤) | HO600 | `offline_scaled_20261008_150131` |
| awfm v2(ε=0、0.3、全局保持、针对性保持) | HO600 | `offline_scaled_20261008_150131/awfm_v2_study_comparison.json`,作业 63863148–63863151 |
| 训练臂官方全量重评 | FULL | 清单记录作业 64539097，正文另记录 64591168；两者是否为重提/后续作业待核实。合并 `merge_policy_eval_shards.py`；**本地尚无已核验完成报告** |

## 4. 其他子集上的结果

| 数字 | 范围 | 来源 |
|---|---|---|
| 选择器 +1.77 / +2.17(HGB / ridge)与第二种子 | SUB | `selector_navtest_v3.json` 等,`docs/project_progress.md` |
| 感知噪声(他车 + 地图) | SUB | `run_perception_noise.sh`,`selector_noise{1,2,3}_{clean,matched}.json` |
| 锚点基座 +0.57 | SUB | `anchor_full_20261013_085604/selector_with_anchor_logprob.json`,作业 63806764–63806769 |
| 世界模型特征块 +0.03 | HO | `submit_wm_pca_features.slurm`,`submit_selector_extras.slurm` |
| 结构化评分器 −0.21 | HO | `submit_selector_struct.slurm`,作业 63754547/548 |
| 梯度引导采样 | SUB 与 36 个难场景 | `run_guidance_dev.sh`,`run_guidance_navtest.sh` |

## 5. 锚点全量（已收到结果，原始文件待独立核验）

| 内容 | 位置 |
|---|---|
| 收集 / 几何 / 选择器 | 作业 64538963 / 64538964 / 64538965、64538966 |
| 输出 | `$X/anchor_navtest_multi_20261021_211730/selector_geom_only.json`、`selector_with_anchor_logprob.json` |

## 6. 论文与图

| 内容 | 位置 |
|---|---|
| 论文稿 | `docs/paper_analysis.tex` |
| 图(PDF 与 PNG)与生成脚本 | `docs/figures/`、`docs/make_figures.py` |
| 所有数字总表 | `docs/results_master.md` |
| 计划与框架 | `docs/paper_plan_A.md` |

## 已知的待核对项

- 第 1 节的作业号取自提交回显,建议用 `sacct` 抽查两三个。
- NAVSIM 引用的作者名与出处凭记忆,投稿前核实。
- Qwen 85.3 权重未公开,只引用其报告值。

## 2026-10-22 文稿校订状态

- `paper_analysis.tex`：主稿；`main.tex` 与 `iclr2025_conference.tex` 保留为历史稿。
- `paper_evidence.json`：图表数字转录与验证状态，图由 `make_figures.py` 读取。尚未替代 HPC 原始 JSON/CSV。
- `paper_revision_20261022.md`：本轮修改和未完成证据清单。
- 旧稿与十张图已完整备份于 `/Users/huamin/Documents/Codex/2026-09-22/zhe/outputs/paper_backup_20261022`。
- 仍须取回全量选择器、地图拆分/TTC/噪声、AWFM 原始报告，及全量训练报告；锚点汇总已收到，但原始输出尚未独立核验；未把排队或提交当作完成。

- 10 月 22 日同步收到 ε=1.0 / 2.0 汇总：相对 SFT 为 +0.016 / +0.086，区间均跨零；原始输出路径与哈希尚需补齐。稿件及图 6 已纳入，未将其列为未运行。

### 跨基座图表同步（2026-10-22）

- `fig11_crossbase_tradeoff.pdf/png`：FULL，两类 HGB 各对独立 SFT；PDMS +2.90 / +1.09，TTC 净修复 −1 / −123。
- 数字见 `results_master.md` B 与 B5；绘图源 `paper_evidence.json` 的 `crossbase`。
- 新增表 `tab:crossbase`；不能用 collection probes 的分项差替换独立 SFT 账本。


## RA-L 全文修订 2026-10-29

- 主稿 paper_ral.tex；已编译 paper_ral.pdf（7页），Overleaf 包 ral_overleaf.zip。
- 数据范围：results_master.md D2/D5/D6/D8；主表12运行（4种子×3规则），配对图8比较。
- PDMS 种子22/23：训练 64939402/64939403，评估 64939404（24分片：0–11=种子22，12–23=种子23），清单 `$X/official_navtest_trained_eval/arms_pdms2223.txt`，脚本 `scripts/hpc/run_pdms_seeds_navtest.sh`。10-29 全部 COMPLETED，合并 12146 场景；种子23 绝对分 86.843。
- 噪声底线与潜变量复现：`run_latent_robustness.sh`（标签 sft_L2、sft_L3、{nearest,margin,pdms}pref_s20_L2）。放大实验：`run_scaleup_navtest.sh`（scaleup_{margin,nearest}pref_s20）。多种子：`run_seeds_navtest.sh`。
- 命令：`merge_policy_eval_shards.py ... --expect-scenes 12146`；`paired_scores_by_log.py BASE.csv ARM.csv token_log_map.json`；`build_ledger.py --group "label|base.csv|arm.csv"`。
- 绘图 make_ral_figures.py + make_ral_result_plots.py；数值转录 figures/ral/ledger_plot_data.json 和 result_plot_data.json。
- 2026-10-30 已重算 12 个运行的原始 CSV（D9）并核对训练配置与共享 manifest 哈希（D11，run.json，12 个运行一致）。各运行实际完成的更新数尚未汇总。

## 2026-11-01 补充实验
- 机制消融 pdms_capped(训练 65068491–94,评估 65068495,860 对):results_master D12;脚本 `scripts/hpc/run_ablation_navtest.sh`,规则在 `tools/rl_prep/build_margin_preferences.py`(`--winner-rule pdms_capped`)。
- 目标侧统计(D13):`tools/rl_prep/target_statistics.py`,输出 `$O/target_statistics.json`。
- 排队中(D14):随机赢家 `run_randwin_navtest.sh`(训练 65196259–62,评估 65196263);共同 860 场景对照 `run_common860_navtest.sh nearest|pdms`(评估 65196832、65196837)。随机规则按 token 哈希取种子(`--selection-seed 0`)。
- 未运行:`tools/rl_prep/measure_fit_error.py`(ε_w,需 GPU,未测试)。
- 评估输出 `validation/<token>.npz` 保存了 relative/absolute 轨迹,EPDMS 可从中重打分(需 NAVSIM v2 缓存与接口,未验证)。
