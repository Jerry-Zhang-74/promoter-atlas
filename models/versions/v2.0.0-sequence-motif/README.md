# v2.0.0-sequence-motif · 序列与 motif 融合 / Sequence and motif fusion

## 中文

在 200 bp 序列分支之外加入 192 维聚合 motif 特征，联合预测 E/S/B。该试验使用 20,000 / 4,000 / 4,000 条训练、验证和测试记录。

与 v1.0 的纯序列模型、v2.1 的纯 motif 模型一起看，可以检查两类输入的贡献；v2.2 则进一步考察缩减 motif 数量。复现需要序列 memmap 和模型卡指定的聚合 motif 表。

### 查看与复现

- [模型卡](cards/promoter_model_v2_pilot_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [训练流程：train_promoter_model_v2_pilot.py](../../../scripts/research/training/train_promoter_model_v2_pilot.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v2.0.0-sequence-motif)

原实验名称：`promoter_model_v2_pilot`。权重源文件时间：`2026-03-15T15:57:05.398602+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

The model combines a 200 bp sequence branch with 192 aggregated motif features to predict E/S/B. The pilot uses 20,000 / 4,000 / 4,000 training, validation and test records.

Compare it with the sequence-only v1.0 and motif-only v2.1 to examine the contributions of both inputs. v2.2 tests a smaller motif feature set. Reproduction uses the sequence memmap and aggregated motif table named in the model card.

### Inspect and reproduce

- [Model card](cards/promoter_model_v2_pilot_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Training procedure: train_promoter_model_v2_pilot.py](../../../scripts/research/training/train_promoter_model_v2_pilot.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v2.0.0-sequence-motif)

Original experiment name: `promoter_model_v2_pilot`. Source checkpoint timestamp: `2026-03-15T15:57:05.398602+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
