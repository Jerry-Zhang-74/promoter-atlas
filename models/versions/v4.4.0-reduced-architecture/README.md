# v4.4.0-reduced-architecture · 16 维结构缩减试验 / Reduced 16-feature architecture model

## 中文

由第二版结构特征筛出 16 维输入，独立预测 E/S/B。特征资源为 `promoter_architecture_features_v4_arch2_reduced.tsv`。

此版检查缩减特征后的效果，直接对照为 18 维的 v4.2。除完整指标外，目录还保存按验证集平均 MSE 选择的最佳结果汇总，便于追踪第 56 个 epoch 的选择。

### 查看与复现

- [模型卡](cards/promoter_model_v4_arch2_reduced_pilot_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v4_arch2_reduced_pilot.py](../../../scripts/research/training/train_promoter_model_v4_arch2_reduced_pilot.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.4.0-reduced-architecture)

原实验名称：`promoter_model_v4_arch2_reduced_pilot`。权重源文件时间：`2026-03-16T13:01:41.806236+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This model predicts E/S/B from a reduced set of 16 second-generation architecture features stored in `promoter_architecture_features_v4_arch2_reduced.tsv`.

The direct comparison is the 18-feature v4.2. Alongside the full metrics, the directory retains the best-result summary selected by validation mean MSE, documenting selection of epoch 56.

### Inspect and reproduce

- [Model card](cards/promoter_model_v4_arch2_reduced_pilot_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v4_arch2_reduced_pilot.py](../../../scripts/research/training/train_promoter_model_v4_arch2_reduced_pilot.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.4.0-reduced-architecture)

Original experiment name: `promoter_model_v4_arch2_reduced_pilot`. Source checkpoint timestamp: `2026-03-16T13:01:41.806236+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
