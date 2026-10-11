# v5.2.0-dual-nobn · 200 bp 与 2000 bp 双流 / 200 bp and 2000 bp dual stream

## 中文

分别编码 200 bp 局部序列和 2000 bp 上游上下文，再融合预测 E/S/B。此版采用省去批归一化的配置，参数量 534,435。

与 long-nobn 对照可检查单独保留局部塔的作用。它也是 v6 扩展为 400 bp 局部窗口、多物种和多任务结构之前的双流试验。

### 查看与复现

- [模型卡](cards/v2_dual_nobn_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_dual.py](../../../scripts/research/training/train_promoter_model_v2_dual.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.2.0-dual-nobn)

原实验名称：`v2_dual_nobn`。权重源文件时间：`2026-09-10T10:06:20.157856+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

Separate towers encode a 200 bp local sequence and 2000 bp upstream context before fusion for E/S/B prediction. This configuration removes batch normalization and has 534,435 parameters.

Compare it with long-nobn to examine the separate local tower. It is also the dual-stream experiment preceding the 400 bp local window, multispecies inputs and multitask architecture of v6.

### Inspect and reproduce

- [Model card](cards/v2_dual_nobn_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_dual.py](../../../scripts/research/training/train_promoter_model_v2_dual.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.2.0-dual-nobn)

Original experiment name: `v2_dual_nobn`. Source checkpoint timestamp: `2026-09-10T10:06:20.157856+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
