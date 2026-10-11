# v3.0.0-pair-grammar · motif 配对特征 / Motif-pair features

## 中文

将 200 bp 序列与 20 维 motif 配对特征一起输入，探索启动子中 motif 组合信息对 E/S/B 预测的作用。对应资源为 `promoter_motif_pair_features_v1.tsv`。

这是从单个 motif 聚合走向轻量组合特征的试验，训练、验证和测试为 20,000 / 4,000 / 4,000 条记录。可以与 v2.0、v2.2 对照，分别检查完整聚合、缩减聚合与配对特征。

### 查看与复现

- [模型卡](cards/promoter_model_v3_pair_pilot_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v3_pair_pilot.py](../../../scripts/research/training/train_promoter_model_v3_pair_pilot.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v3.0.0-pair-grammar)

原实验名称：`promoter_model_v3_pair_pilot`。权重源文件时间：`2026-03-15T17:18:46.098737+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This pilot combines the 200 bp sequence with 20 motif-pair features to explore motif combinations in E/S/B prediction. Its feature resource is `promoter_motif_pair_features_v1.tsv`.

It moves from individual motif aggregates toward lightweight combination features, using 20,000 / 4,000 / 4,000 training, validation and test records. Compare it with v2.0 and v2.2 to examine full aggregates, reduced aggregates and pair features.

### Inspect and reproduce

- [Model card](cards/promoter_model_v3_pair_pilot_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v3_pair_pilot.py](../../../scripts/research/training/train_promoter_model_v3_pair_pilot.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v3.0.0-pair-grammar)

Original experiment name: `promoter_model_v3_pair_pilot`. Source checkpoint timestamp: `2026-03-15T17:18:46.098737+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
