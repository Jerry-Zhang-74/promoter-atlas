# v2.1.0-motif-only · motif 独立对照 / Motif-only control

## 中文

仅输入 192 维聚合 motif 特征，由特征分支预测 E/S/B。保留此版是为了回答：v2.0 融合模型中的信号有多少能够由 motif 特征单独解释。

训练、验证和测试分别为 20,000 / 4,000 / 4,000 条记录。比较时同时查看 v2.0 和 v1.0 的逐轴指标，模型卡给出了特征表和缺失 S 标签的处理方式。

### 查看与复现

- [模型卡](cards/motif_only_pilot_v1_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_motif_only_pilot_v1.py](../../../scripts/research/training/train_motif_only_pilot_v1.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v2.1.0-motif-only)

原实验名称：`motif_only_pilot_v1`。权重源文件时间：`2026-03-15T16:58:03.631854+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This control predicts E/S/B from 192 aggregated motif features alone. Its purpose is to examine how much of the signal in the v2.0 fusion model can be captured by the motif branch itself.

Training, validation and test contain 20,000 / 4,000 / 4,000 records. Compare its per-axis metrics with v2.0 and v1.0. The card identifies the feature table and masked treatment of missing S labels.

### Inspect and reproduce

- [Model card](cards/motif_only_pilot_v1_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_motif_only_pilot_v1.py](../../../scripts/research/training/train_motif_only_pilot_v1.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v2.1.0-motif-only)

Original experiment name: `motif_only_pilot_v1`. Source checkpoint timestamp: `2026-03-15T16:58:03.631854+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
