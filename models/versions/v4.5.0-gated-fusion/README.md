# v4.5.0-gated-fusion · 门控融合与 E–S 去相关 / Gated fusion and E–S decorrelation

## 中文

让门控模块学习 200 bp 序列表示与 18 维结构特征的融合权重，再预测 E/S/B。训练额外使用权重 0.02 的 E–S 去相关项。

与 v4.1 的融合路线对照，可检查门控和去相关目标组合后的结果。评估表同时记录逐轴精度及预测 E、S 的相关诊断；最佳结果汇总对应验证平均 MSE 选中的 epoch 91。

### 查看与复现

- [模型卡](cards/promoter_model_v4_fusion_pilot_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v4_fusion_pilot.py](../../../scripts/research/training/train_promoter_model_v4_fusion_pilot.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.5.0-gated-fusion)

原实验名称：`promoter_model_v4_fusion_pilot`。权重源文件时间：`2026-03-16T22:48:17.214337+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

A learned gate weights the 200 bp sequence representation and 18 architecture features before E/S/B prediction. Training also uses an E–S decorrelation term with weight 0.02.

Compare this experiment with the fusion route in v4.1 to examine the combined gate and decorrelation objective. Evaluations include per-axis accuracy and correlation diagnostics for predicted E and S; the best-result summary records epoch 91 selected by validation mean MSE.

### Inspect and reproduce

- [Model card](cards/promoter_model_v4_fusion_pilot_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v4_fusion_pilot.py](../../../scripts/research/training/train_promoter_model_v4_fusion_pilot.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.5.0-gated-fusion)

Original experiment name: `promoter_model_v4_fusion_pilot`. Source checkpoint timestamp: `2026-03-16T22:48:17.214337+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
