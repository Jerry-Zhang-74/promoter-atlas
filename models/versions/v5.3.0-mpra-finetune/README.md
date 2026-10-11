# v5.3.0-mpra-finetune · K562 MPRA 全网微调 / Whole-network fine-tuning for K562 MPRA

## 中文

以 baseline-repro 的 GTEx 预训练卷积编码器初始化，替换为单输出活性头并微调整个网络。输入为 GSE118242 K562 实验的 164 bp 合成核心启动子，输出 RNA/DNA 比值的 Mean_log10 活性。

三个版本的差别是编码器初始化与更新方式。比较时同时查看测试序列相关性和家族内变体排序；对应指标保存在本目录 evaluation 中。

### 查看与复现

- [模型卡](cards/mpra_finetune_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_mpra_finetune.py](../../../scripts/research/training/train_mpra_finetune.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.3.0-mpra-finetune)

原实验名称：`mpra_finetune`。权重源文件时间：`2026-09-10T17:24:41.431867+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This starts from the GTEx-pretrained baseline-repro convolutional encoder, replaces the head with a scalar activity output, and fine-tunes the network. Inputs are 164 bp synthetic core promoters from the GSE118242 K562 assay; the output is Mean_log10 activity of the RNA/DNA ratio.

The three variants differ in encoder initialization and updating. Compare both test-sequence correlation and within-family variant ranking using the metrics in this directory’s evaluation folder.

### Inspect and reproduce

- [Model card](cards/mpra_finetune_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_mpra_finetune.py](../../../scripts/research/training/train_mpra_finetune.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.3.0-mpra-finetune)

Original experiment name: `mpra_finetune`. Source checkpoint timestamp: `2026-09-10T17:24:41.431867+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
