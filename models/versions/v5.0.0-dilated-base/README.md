# v5.0.0-dilated-base · 膨胀残差卷积基线 / Dilated residual baseline

## 中文

使用 200 bp 序列驱动膨胀残差卷积网络，预测 E/S/B，参数量 491,747。该分支改变序列编码器结构。

与 baseline-repro 比较卷积骨架。评估和日志保留了此版训练结果。

### 查看与复现

- [模型卡](cards/v2_dilated_base_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_distill.py](../../../scripts/research/training/train_promoter_model_v2_distill.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.0.0-dilated-base)

原实验名称：`v2_dilated_base`。权重源文件时间：`2026-09-10T01:13:28.451851+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

A dilated residual convolutional network predicts E/S/B from 200 bp sequences, with 491,747 parameters. This branch explores the sequence-encoder architecture.

Compare it with baseline-repro for the convolutional backbone. Its evaluation and log retain the training results.

### Inspect and reproduce

- [Model card](cards/v2_dilated_base_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_distill.py](../../../scripts/research/training/train_promoter_model_v2_distill.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.0.0-dilated-base)

Original experiment name: `v2_dilated_base`. Source checkpoint timestamp: `2026-09-10T01:13:28.451851+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
