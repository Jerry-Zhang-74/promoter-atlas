# v5.0.0-baseline-repro · 卷积基线复跑 / Convolutional baseline rerun

## 中文

在后续统一训练流程中复跑 200 bp 纯序列 ConvBaseline，参数量 207,939，输出 E/S/B。它也是 v5.3 MPRA 迁移试验的预训练来源。

用它对照 v5.0 的特征融合、膨胀卷积和蒸馏变体，可以逐项追踪该阶段的模型设计；模型原名 `v2_baseline_repro` 保留在卡片与日志中。

### 查看与复现

- [模型卡](cards/v2_baseline_repro_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_distill.py](../../../scripts/research/training/train_promoter_model_v2_distill.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.0.0-baseline-repro)

原实验名称：`v2_baseline_repro`。权重源文件时间：`2026-09-10T00:56:20.570762+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This reruns the 200 bp sequence-only ConvBaseline in the later training pipeline. It has 207,939 parameters and E/S/B outputs, and provides the pretrained backbone for the v5.3 MPRA transfer experiments.

Use it as the reference for the v5.0 feature-fusion, dilated-convolution and distillation variants. Its original name, `v2_baseline_repro`, is retained in the card and log.

### Inspect and reproduce

- [Model card](cards/v2_baseline_repro_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_distill.py](../../../scripts/research/training/train_promoter_model_v2_distill.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.0.0-baseline-repro)

Original experiment name: `v2_baseline_repro`. Source checkpoint timestamp: `2026-09-10T00:56:20.570762+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
