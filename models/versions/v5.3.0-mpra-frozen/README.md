# v5.3.0-mpra-frozen · K562 MPRA 冻结迁移 / Frozen-backbone transfer to K562 MPRA

## 中文

固定 baseline-repro 的 GTEx 预训练序列编码器，只训练新的单输出头，检查已有表示对报告基因活性的迁移能力。输入为 GSE118242 K562 实验的 164 bp 合成核心启动子，输出 RNA/DNA 比值的 Mean_log10 活性。

三个版本的差别是编码器初始化与更新方式。比较时同时查看测试序列相关性和家族内变体排序；对应指标保存在本目录 evaluation 中。

### 查看与复现

- [模型卡](cards/mpra_frozen_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [三种 MPRA 训练模式](../../../scripts/research/training/train_mpra_finetune.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.3.0-mpra-frozen)

原实验名称：`mpra_frozen`。权重源文件时间：`2026-09-10T17:25:29.974996+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This freezes the GTEx-pretrained sequence encoder from baseline-repro and trains a new scalar head to examine transfer of its existing representation to reporter activity. Inputs are 164 bp synthetic core promoters from the GSE118242 K562 assay; the output is Mean_log10 activity of the RNA/DNA ratio.

The three variants differ in encoder initialization and updating. Compare both test-sequence correlation and within-family variant ranking using the metrics in this directory’s evaluation folder.

### Inspect and reproduce

- [Model card](cards/mpra_frozen_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Three MPRA training modes](../../../scripts/research/training/train_mpra_finetune.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.3.0-mpra-frozen)

Original experiment name: `mpra_frozen`. Source checkpoint timestamp: `2026-09-10T17:25:29.974996+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
