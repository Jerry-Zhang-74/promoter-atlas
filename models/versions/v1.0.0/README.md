# v1.0.0 · 200 bp 序列基线 / 200 bp sequence baseline

## 中文

从启动子上游 200 bp 序列预测 Score_E、Score_S 和 Score_B，是后续加入 motif、序列结构特征和长上下文的起点。S 缺失记录在损失中按掩码处理。

本版适合重看早期纯序列模型的表现；与 v2.0 对照可以检查加入 motif 特征后的变化。目录同时保留原评估与统一重评估，后者补充了预测 E 与 S 的相关性诊断。

### 查看与复现

- [模型卡](cards/promoter_model_v1_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v1.py](../../../scripts/research/training/train_promoter_model_v1.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v1.0.0)

原实验名称：`promoter_model_v1`。权重源文件时间：`2026-03-15T13:02:55.776448+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This baseline predicts Score_E, Score_S and Score_B from the 200 bp upstream promoter sequence. Missing S targets are masked in the loss. It provides the starting point for motif features, promoter-architecture features and longer sequence context.

Use it to revisit the early sequence-only model and compare with v2.0 when examining the addition of motifs. Both the original evaluation and a unified reevaluation are retained; the latter adds diagnostics for correlation between predicted E and S.

### Inspect and reproduce

- [Model card](cards/promoter_model_v1_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v1.py](../../../scripts/research/training/train_promoter_model_v1.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v1.0.0)

Original experiment name: `promoter_model_v1`. Source checkpoint timestamp: `2026-03-15T13:02:55.776448+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
