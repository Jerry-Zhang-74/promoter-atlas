# v4.2.0-architecture-only · 18 维结构独立对照 / 18-feature architecture control

## 中文

直接由 18 维启动子结构特征预测 E/S/B，用于拆解 v4.1 中结构分支的作用。输入表为 `promoter_architecture_features_v2.tsv`。

这个版本适合做结构特征对照。后续 v4.3 扩展特征，v4.4 缩减特征，v4.6 在同一类输入上试验 S 残差建模。

### 查看与复现

- [模型卡](cards/promoter_model_v4_arch_only_pilot_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v4_arch_only_pilot.py](../../../scripts/research/training/train_promoter_model_v4_arch_only_pilot.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.2.0-architecture-only)

原实验名称：`promoter_model_v4_arch_only_pilot`。权重源文件时间：`2026-03-15T23:02:36.470202+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This control predicts E/S/B directly from the 18 promoter-architecture features in `promoter_architecture_features_v2.tsv`. It isolates the feature branch used by v4.1.

Use it as the architecture-only reference: v4.3 expands the features, v4.4 reduces them, and v4.6 investigates residual modeling of S using the same input family.

### Inspect and reproduce

- [Model card](cards/promoter_model_v4_arch_only_pilot_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v4_arch_only_pilot.py](../../../scripts/research/training/train_promoter_model_v4_arch_only_pilot.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.2.0-architecture-only)

Original experiment name: `promoter_model_v4_arch_only_pilot`. Source checkpoint timestamp: `2026-03-15T23:02:36.470202+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
