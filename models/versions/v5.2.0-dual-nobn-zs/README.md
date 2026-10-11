# v5.2.0-dual-nobn-zs · 双流模型与 S 标准化 / Dual stream with standardized S

## 中文

保留 200 bp + 2000 bp 双流结构，对 S 训练目标按训练集统计量标准化，评估时恢复尺度。模型参数量 534,435，输出 E/S/B。

与 dual-nobn 比较 S 标准化，与 long-nobn-zs 比较单塔和双塔输入。这样可以分别检查目标尺度与序列结构两项设计。

### 查看与复现

- [模型卡](cards/v2_dual_nobn_zs_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_dual.py](../../../scripts/research/training/train_promoter_model_v2_dual.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.2.0-dual-nobn-zs)

原实验名称：`v2_dual_nobn_zs`。权重源文件时间：`2026-09-10T11:00:13.807596+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This retains the 200 bp + 2000 bp dual-stream structure and standardizes S using TRAIN statistics, restoring its scale during evaluation. It has 534,435 parameters and predicts E/S/B.

Compare with dual-nobn for S standardization and with long-nobn-zs for the single- versus dual-tower design, examining target scaling and sequence architecture separately.

### Inspect and reproduce

- [Model card](cards/v2_dual_nobn_zs_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_dual.py](../../../scripts/research/training/train_promoter_model_v2_dual.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.2.0-dual-nobn-zs)

Original experiment name: `v2_dual_nobn_zs`. Source checkpoint timestamp: `2026-09-10T11:00:13.807596+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
