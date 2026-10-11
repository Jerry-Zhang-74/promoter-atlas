# v5.0.0-conv-both · 卷积与结构、motif 联合特征 / Convolution with architecture and motif features

## 中文

将 16 维结构特征和 64 维 motif 特征合成 80 维向量，与 200 bp 序列卷积表示拼接后预测 E/S/B。模型参数量 218,179。

此版用于查看两类显式特征共同参与预测的结果。与 conv-arch 对照 motif 输入，与 baseline-repro 对照整条显式特征分支。

### 查看与复现

- [模型卡](cards/v2_conv_both_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_fusion.py](../../../scripts/research/training/train_promoter_model_v2_fusion.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.0.0-conv-both)

原实验名称：`v2_conv_both`。权重源文件时间：`2026-09-10T01:49:57.803682+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This model concatenates 16 architecture features and 64 motif features into an 80-dimensional vector, joins it to the 200 bp convolutional representation, and predicts E/S/B. It has 218,179 parameters.

Use it to examine the combined explicit-feature branch. conv-arch provides the comparison for adding motifs, and baseline-repro provides the sequence-only reference.

### Inspect and reproduce

- [Model card](cards/v2_conv_both_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_fusion.py](../../../scripts/research/training/train_promoter_model_v2_fusion.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.0.0-conv-both)

Original experiment name: `v2_conv_both`. Source checkpoint timestamp: `2026-09-10T01:49:57.803682+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
