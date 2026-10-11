# v4.3.0-architecture · 32 维结构特征 / 32-feature architecture model

## 中文

使用第三版启动子结构表 `promoter_architecture_features_v3.tsv` 的 32 维特征预测 E/S/B。该试验沿结构特征单独建模的路线扩充输入。

与 v4.2 的 18 维版本对照，可检查增加特征后的逐轴表现；v4.4 提供缩减输入的另一端对照。复现时根据本版模型卡选择第三版特征表。

### 查看与复现

- [模型卡](cards/promoter_model_v4_arch3_pilot_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v4_arch3_pilot.py](../../../scripts/research/training/train_promoter_model_v4_arch3_pilot.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.3.0-architecture)

原实验名称：`promoter_model_v4_arch3_pilot`。权重源文件时间：`2026-03-15T23:16:05.738280+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This experiment predicts E/S/B using 32 features from `promoter_architecture_features_v3.tsv`, extending the architecture-only route.

Compare its per-axis performance with the 18-feature v4.2; v4.4 supplies a reduced-input comparison. Reproduction should use the third-generation feature table identified by this model card.

### Inspect and reproduce

- [Model card](cards/promoter_model_v4_arch3_pilot_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v4_arch3_pilot.py](../../../scripts/research/training/train_promoter_model_v4_arch3_pilot.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.3.0-architecture)

Original experiment name: `promoter_model_v4_arch3_pilot`. Source checkpoint timestamp: `2026-03-15T23:16:05.738280+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
