# v5.0.0-conv-arch · 卷积与结构特征 / Convolution with architecture features

## 中文

在 200 bp 卷积序列表示上拼接 16 维结构特征，预测 E/S/B；模型卡记录参数量 209,987。

与 baseline-repro 对照结构输入的加入，与 conv-both 对照增加 motif 输入后的变化。

### 查看与复现

- [模型卡](cards/v2_conv_arch_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_fusion.py](../../../scripts/research/training/train_promoter_model_v2_fusion.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.0.0-conv-arch)

原实验名称：`v2_conv_arch`。权重源文件时间：`2026-09-10T01:41:12.508048+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This variant appends 16 architecture features to the 200 bp convolutional representation and predicts E/S/B, with 209,987 parameters recorded in the card.

Compare it with baseline-repro for the architecture input and conv-both for added motifs.

### Inspect and reproduce

- [Model card](cards/v2_conv_arch_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_fusion.py](../../../scripts/research/training/train_promoter_model_v2_fusion.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.0.0-conv-arch)

Original experiment name: `v2_conv_arch`. Source checkpoint timestamp: `2026-09-10T01:41:12.508048+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
