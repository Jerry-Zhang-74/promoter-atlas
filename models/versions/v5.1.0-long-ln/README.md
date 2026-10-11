# v5.1.0-long-ln · 长上下文的组归一化试验 / Group normalization in the long-context model

## 中文

由 2000 bp 序列预测 E/S/B，采用历史参数名 `--norm ln`。对应训练源码实际构造的是 `GroupNorm(8, channels)`，复现时应沿用这一实现。参数量为 314,627。

它与 long-base、long-nobn 一起比较批归一化、组归一化和省去归一化层这三种配置。模型卡和日志沿用原名 `v2_long_ln`。

### 查看与复现

- [模型卡](cards/v2_long_ln_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_longcontext.py](../../../scripts/research/training/train_promoter_model_v2_longcontext.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.1.0-long-ln)

原实验名称：`v2_long_ln`。权重源文件时间：`2026-09-10T09:39:41.030607+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This model predicts E/S/B from 2000 bp sequences using the historical option `--norm ln`. The training source implements that option as `GroupNorm(8, channels)`, which is the implementation to use for reproduction. It has 314,627 parameters.

Together with long-base and long-nobn, it compares batch normalization, group normalization and removal of normalization layers. The card and log retain the original name `v2_long_ln`.

### Inspect and reproduce

- [Model card](cards/v2_long_ln_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_longcontext.py](../../../scripts/research/training/train_promoter_model_v2_longcontext.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.1.0-long-ln)

Original experiment name: `v2_long_ln`. Source checkpoint timestamp: `2026-09-10T09:39:41.030607+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
