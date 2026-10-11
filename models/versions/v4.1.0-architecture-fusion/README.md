# v4.1.0-architecture-fusion · 18 维结构与序列融合 / Sequence fusion with 18 architecture features

## 中文

沿用 200 bp 序列分支，将 v4.0 的结构输入扩展为 18 维 `promoter_architecture_features_v2.tsv`，预测 E/S/B。

与 v4.0 比较结构特征集的扩展，与 v4.2 比较序列分支的贡献。模型卡记录了 20,000 / 4,000 / 4,000 的试验划分，评估表保存逐轴结果。

### 查看与复现

- [模型卡](cards/promoter_model_v4_arch2_pilot_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v4_arch2_pilot.py](../../../scripts/research/training/train_promoter_model_v4_arch2_pilot.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.1.0-architecture-fusion)

原实验名称：`promoter_model_v4_arch2_pilot`。权重源文件时间：`2026-03-15T22:44:11.083556+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This model retains the 200 bp sequence branch and expands the v4.0 architecture input to the 18 features in `promoter_architecture_features_v2.tsv`, predicting E/S/B.

Compare it with v4.0 for the feature-set expansion and with v4.2 for the contribution of the sequence branch. The card records the 20,000 / 4,000 / 4,000 pilot split, and the evaluation table provides per-axis results.

### Inspect and reproduce

- [Model card](cards/promoter_model_v4_arch2_pilot_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v4_arch2_pilot.py](../../../scripts/research/training/train_promoter_model_v4_arch2_pilot.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.1.0-architecture-fusion)

Original experiment name: `promoter_model_v4_arch2_pilot`. Source checkpoint timestamp: `2026-03-15T22:44:11.083556+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
