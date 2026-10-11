# 基因模型卡 v6.15.1-release.rc2 · Gene model card

[中文](#中文) · [English](#english)

## 中文

该版为 `v6.15.1-release.rc2` 实验候选，沿用 batch16 `v6.15.0-multitask.rc1` 权重，整理打包和来源记录。rc2 还修复 F17-3 空 batch 的类型化输入错误。

### 模型与选择

任务是人类基因级自然表达 E/S/B 回归。独立臂 A 由三个单任务 CNN 构成，每个 217,857 参数，输入为 4 条启动子序列与固定 1069 维组成/位置特征分支。发布 seed=20261016，seed=20261017 用于重复实验。

选择规则为验证集平均逐轴 rho 增益 ≥0.02 且任一轴下降 ≤0.02；共享臂 B/C 的平均增益为 -0.0061/-0.0049，最终选择独立臂，详见 `negative_ablation_results.json`。

### 指标与研究范围

| 输出 | Spearman | R² | MAE | n |
|---|---:|---:|---:|---:|
| E | 0.7428 | 0.5618 | 0.1513 | 5851 |
| S | 0.4341 | 0.1698 | 0.2066 | 1880 |
| B | 0.7262 | 0.6030 | 0.1786 | 5851 |

两 seed 平均 TEST E/S/B 为 0.7411/0.4273/0.7215；预测均值集成为 0.7459/0.4335/0.7252；匹配 HGB 为 0.7335/0.4334/0.7191，S 与神经模型统计上相当。

TEST 是开发阶段已接触过基因的回顾性留出集。A 主 seed 及 A 第二 seed 的 E/S 用原训练器全局 RNG shuffle，B/C 和 A 第二 seed 的 B 用恢复训练器逐 epoch 确定性采样；初始化 RNG 和数据顺序差异影响共享任务比较。辅助表观数据以 0 编码未覆盖测定，其缺失语义待解决。发现的调控家族覆盖 14 个构建体。该版 component-union 划分与早期随机划分分别报告。

适用输入是人类基因的真实 4-promoter bag 与 1069 维特征，S 为可检出组织的负 MAD。片段/合成构建体活性、HEK293T、其他物种、组织/转录本特异预测与时间稳定性需要对应的数据和评估。

---

## English

This `v6.15.1-release.rc2` experimental candidate retains batch16 `v6.15.0-multitask.rc1` weights while repairing packaging and provenance. rc2 also repairs F17-3 typed validation for empty batches.

### Model and selection

The task is human gene-level natural-expression E/S/B regression. Independent arm A comprises three single-task CNNs, each with 217,857 parameters, using four promoter sequences and a fixed 1069-dimensional composition/position branch. The shipped seed is 20261016; seed 20261017 supports replication.

The validation rule requires mean per-axis rho gain ≥0.02 and every axis drop ≤0.02. Shared arms B/C had mean gains -0.0061/-0.0049; the independent arm was selected. See `negative_ablation_results.json`.

### Metrics and research domain

| Head | Spearman | R² | MAE | n |
|---|---:|---:|---:|---:|
| E | 0.7428 | 0.5618 | 0.1513 | 5851 |
| S | 0.4341 | 0.1698 | 0.2066 | 1880 |
| B | 0.7262 | 0.6030 | 0.1786 | 5851 |

Mean-of-seed TEST E/S/B is 0.7411/0.4273/0.7215; mean-prediction ensemble is 0.7459/0.4335/0.7252; matched HGB is 0.7335/0.4334/0.7191, with S statistically tied with the neural model.

TEST is retrospective and its genes were visible during development. Reference A and seed-2 A E/S used the original global-RNG shuffle trainer; B/C and seed-2 A B used a recovery trainer with deterministic per-epoch sampling. Initialization RNG and data-order differences confound task-sharing contrasts. Uncovered auxiliary assays were encoded as zero, leaving missingness unresolved. Discovered regulatory families cover 14 constructs. Report this component-union split separately from earlier random splits.

Applicable inputs are genuine four-promoter bags and 1069-dimensional features for human genes; S is negative MAD across detectable tissues. Fragment/synthetic activity, HEK293T, other species, tissue/isoform-specific prediction, and temporal stability require corresponding data and evaluations.
