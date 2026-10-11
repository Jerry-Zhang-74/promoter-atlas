# v4.6.0-residual-s · S 的残差建模 / Residual modeling of S

## 中文

输入为 18 维结构特征。先在训练集拟合 S 与 E 的线性关系，再让模型学习 S 的剩余部分；评估时按 `S = a × 预测E + b + 预测残差` 还原 S，E 与 B 仍分别预测。

这项试验考察 E–S 耦合的处理方式，可与 v4.2 和 v4.5 对照。模型卡保存 a、b 系数，评估表包含还原后 S 与残差诊断。标签定义继续见项目标签规范。

### 查看与复现

- [模型卡](cards/promoter_model_v4_residual_s_pilot_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v4_residual_s_pilot.py](../../../scripts/research/training/train_promoter_model_v4_residual_s_pilot.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.6.0-residual-s)

原实验名称：`promoter_model_v4_residual_s_pilot`。权重源文件时间：`2026-03-17T11:30:38.484770+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

The input is an 18-feature architecture vector. A linear relationship between S and E is fitted on TRAIN, and the model learns the remaining S residual. Evaluation reconstructs `S = a × predicted E + b + predicted residual`, alongside separate E and B predictions.

The experiment examines treatment of E–S coupling and can be compared with v4.2 and v4.5. The card stores a and b; evaluation includes reconstructed S and residual diagnostics. The project label specification remains the reference for target definitions.

### Inspect and reproduce

- [Model card](cards/promoter_model_v4_residual_s_pilot_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v4_residual_s_pilot.py](../../../scripts/research/training/train_promoter_model_v4_residual_s_pilot.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v4.6.0-residual-s)

Original experiment name: `promoter_model_v4_residual_s_pilot`. Source checkpoint timestamp: `2026-03-17T11:30:38.484770+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
