# v5.1.0-long-nobn · 省去归一化层的长上下文模型 / Long-context model without normalization layers

## 中文

输入为上游 2000 bp，膨胀残差编码器中的归一化位置使用恒等映射，输出 E/S/B。模型参数量为 312,579。

这是 long-base 的归一化消融，也是 long-nobn-zs 的直接基线；后者在此骨架上调整 S 目标的训练尺度。

### 查看与复现

- [模型卡](cards/v2_long_nobn_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_longcontext.py](../../../scripts/research/training/train_promoter_model_v2_longcontext.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.1.0-long-nobn)

原实验名称：`v2_long_nobn`。权重源文件时间：`2026-09-10T08:59:23.075276+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

The model takes 2000 upstream bases and uses identity mappings at the normalization positions in the dilated residual encoder. It predicts E/S/B and has 312,579 parameters.

It is the normalization ablation for long-base and the direct reference for long-nobn-zs, which changes the S training-target scale on this backbone.

### Inspect and reproduce

- [Model card](cards/v2_long_nobn_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_longcontext.py](../../../scripts/research/training/train_promoter_model_v2_longcontext.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.1.0-long-nobn)

Original experiment name: `v2_long_nobn`. Source checkpoint timestamp: `2026-09-10T08:59:23.075276+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
