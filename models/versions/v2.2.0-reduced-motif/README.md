# v2.2.0-reduced-motif · 缩减 motif 特征 / Reduced motif features

## 中文

保留 200 bp 序列分支，将 v2.0 的 192 维 motif 聚合输入缩减为 64 维，输出仍为 E/S/B。训练、验证和测试规模沿用 20,000 / 4,000 / 4,000。

此版用于比较 motif 特征数量与模型表现的关系。复现时使用 `promoter_motif_reduced_v1.tsv`；与完整 motif 融合版 v2.0 配对阅读最直接。

### 查看与复现

- [模型卡](cards/promoter_model_v2_reduced_pilot_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_reduced_pilot.py](../../../scripts/research/training/train_promoter_model_v2_reduced_pilot.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v2.2.0-reduced-motif)

原实验名称：`promoter_model_v2_reduced_pilot`。权重源文件时间：`2026-03-15T17:10:18.276481+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

The 200 bp sequence branch is retained while the 192 motif features used by v2.0 are reduced to 64. Outputs remain E/S/B, with 20,000 / 4,000 / 4,000 training, validation and test records.

Use this experiment to compare feature-set size with predictive performance. Reproduction uses `promoter_motif_reduced_v1.tsv`; the direct comparison is the full motif fusion model v2.0.

### Inspect and reproduce

- [Model card](cards/promoter_model_v2_reduced_pilot_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_reduced_pilot.py](../../../scripts/research/training/train_promoter_model_v2_reduced_pilot.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v2.2.0-reduced-motif)

Original experiment name: `promoter_model_v2_reduced_pilot`. Source checkpoint timestamp: `2026-03-15T17:10:18.276481+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
