# v5.1.0-long-both · 长上下文与显式特征 / Long context with explicit features

## 中文

在 2000 bp 单塔表示上拼接 80 维结构与 motif 特征，预测 E/S/B，参数量 324,867。

与 long-base 比较显式特征在长窗口模型中的贡献，也可以与 200 bp 的 conv-both 对照输入窗口变化。所需输入包括 2000 bp 序列和对应的结构、motif 表。

### 查看与复现

- [模型卡](cards/v2_long_both_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_longcontext.py](../../../scripts/research/training/train_promoter_model_v2_longcontext.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.1.0-long-both)

原实验名称：`v2_long_both`。权重源文件时间：`2026-09-10T08:30:36.680589+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This model appends 80 architecture and motif features to the 2000 bp single-tower representation and predicts E/S/B. It has 324,867 parameters.

Compare with long-base for explicit features in the long-window model, and with the 200 bp conv-both for the change in sequence window. Inputs comprise 2000 bp sequences and the associated architecture and motif tables.

### Inspect and reproduce

- [Model card](cards/v2_long_both_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_longcontext.py](../../../scripts/research/training/train_promoter_model_v2_longcontext.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.1.0-long-both)

Original experiment name: `v2_long_both`. Source checkpoint timestamp: `2026-09-10T08:30:36.680589+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
