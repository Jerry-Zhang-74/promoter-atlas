# v5.1.0-long-nobn-zs · 长上下文与 S 标准化 / Long context with standardized S

## 中文

沿用 long-nobn 的 2000 bp 编码器，在训练集上对 S 目标做 z-score 标准化，评估时还原尺度。参数量保持 312,579，仍预测 E/S/B。

这项试验针对 S 目标方差较小的问题；与 long-nobn 对照时，应一并查看逐轴指标和训练日志。复现需要保持训练集统计量与评估还原步骤一致。

### 查看与复现

- [模型卡](cards/v2_long_nobn_zs_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_longcontext.py](../../../scripts/research/training/train_promoter_model_v2_longcontext.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.1.0-long-nobn-zs)

原实验名称：`v2_long_nobn_zs`。权重源文件时间：`2026-09-10T10:33:31.892241+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This version retains the long-nobn 2000 bp encoder, z-scores S using TRAIN statistics, and reverses that scaling during evaluation. It keeps 312,579 parameters and E/S/B outputs.

The experiment addresses the low variance of the S target. Read its per-axis metrics and training log alongside long-nobn, preserving the TRAIN statistics and inverse transformation during reproduction.

### Inspect and reproduce

- [Model card](cards/v2_long_nobn_zs_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_longcontext.py](../../../scripts/research/training/train_promoter_model_v2_longcontext.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.1.0-long-nobn-zs)

Original experiment name: `v2_long_nobn_zs`. Source checkpoint timestamp: `2026-09-10T10:33:31.892241+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
