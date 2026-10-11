# v6.0.0 · 首轮多物种多任务模型 / First multispecies multitask model

## 中文

将局部窗口扩展为 400 bp，与 2000 bp 上下文和物种编号共同输入；共享表示同时学习 E/S/B、组学目标、TF 标签及辅助任务。模型卡记录 4,754,932 个参数、400 个 TF 标签和 8 个组学目标。

本目录保留第一轮实现与结果。后续审查定位到 TF 矩阵读取偏移及掩码输入影响主任务评估的问题，修订实现见 v6.1。比较两个版本时可沿源码快照检查这些具体变化。

### 查看与复现

- [模型卡](cards/promoter_atlas_v6_round1_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [该轮源码快照](source/)
- [训练流程：trainer_v6_round1.py](source/trainer_v6_round1.py)
- [训练流程：train_promoter_atlas_v6.py](../../../scripts/research/training/train_promoter_atlas_v6.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v6.0.0)

原实验名称：`promoter_atlas_v6_round1`。权重源文件时间：`2026-09-13T17:33:37.507472+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This first multispecies round combines a 400 bp local window, 2000 bp context and species identifier. A shared representation learns E/S/B, omics targets, TF labels and auxiliary tasks. The card records 4,754,932 parameters, 400 TF labels and eight omics targets.

The directory preserves the first-round implementation and results. Subsequent review identified a TF-matrix loading offset and masked inputs affecting primary-task evaluation; v6.1 contains the revisions. Source snapshots allow those changes to be traced directly.

### Inspect and reproduce

- [Model card](cards/promoter_atlas_v6_round1_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Round-specific source snapshot](source/)
- [Training procedure: trainer_v6_round1.py](source/trainer_v6_round1.py)
- [Training procedure: train_promoter_atlas_v6.py](../../../scripts/research/training/train_promoter_atlas_v6.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v6.0.0)

Original experiment name: `promoter_atlas_v6_round1`. Source checkpoint timestamp: `2026-09-13T17:33:37.507472+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
