# v5.1.0-long-base · 2000 bp 单塔基线 / 2000 bp single-tower baseline

## 中文

将单条输入窗口扩展到上游 2000 bp，用步幅卷积和膨胀残差块编码后预测 E/S/B。此版使用批归一化，参数量 314,627。

该分支用于检查更长上下文的表现。long-both 加入显式特征，long-ln 和 long-nobn 分别调整归一化配置；这些变体可以围绕此基线比较。

### 查看与复现

- [模型卡](cards/v2_long_base_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_longcontext.py](../../../scripts/research/training/train_promoter_model_v2_longcontext.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.1.0-long-base)

原实验名称：`v2_long_base`。权重源文件时间：`2026-09-10T08:02:27.556206+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

The single input window extends to 2000 upstream bases. Strided convolutions and dilated residual blocks encode it for E/S/B prediction. This baseline uses batch normalization and has 314,627 parameters.

The branch examines longer context. long-both adds explicit features, while long-ln and long-nobn change normalization; use this version as their reference.

### Inspect and reproduce

- [Model card](cards/v2_long_base_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_longcontext.py](../../../scripts/research/training/train_promoter_model_v2_longcontext.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.1.0-long-base)

Original experiment name: `v2_long_base`. Source checkpoint timestamp: `2026-09-10T08:02:27.556206+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
