# v4.0.0-architecture-fusion · 首版序列结构融合 / Initial promoter-architecture fusion

## 中文

把 200 bp 序列表示与 6 维启动子结构特征融合后预测 E/S/B，结构输入来自 `promoter_architecture_features_v1.tsv`。这里的“结构”指从启动子序列提取的特征。

此版是 v4 特征路线的起点。v4.1 将结构表扩展到 18 维，v4.2 则单独使用结构特征，适合连起来检查输入设计的变化。

### 查看与复现

- [模型卡](cards/promoter_model_v4_arch_pilot_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v4_arch_pilot.py](../../../scripts/research/training/train_promoter_model_v4_arch_pilot.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.0.0-architecture-fusion)

原实验名称：`promoter_model_v4_arch_pilot`。权重源文件时间：`2026-03-15T22:07:08.842940+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

The model predicts E/S/B by combining a 200 bp sequence representation with six promoter-architecture features from `promoter_architecture_features_v1.tsv`. Architecture here refers to features extracted from promoter sequences.

This is the starting point of the v4 feature series. v4.1 expands the table to 18 features, while v4.2 uses those features on their own; read the three together to follow the input-design experiments.

### Inspect and reproduce

- [Model card](cards/promoter_model_v4_arch_pilot_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v4_arch_pilot.py](../../../scripts/research/training/train_promoter_model_v4_arch_pilot.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.0.0-architecture-fusion)

Original experiment name: `promoter_model_v4_arch_pilot`. Source checkpoint timestamp: `2026-03-15T22:07:08.842940+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
