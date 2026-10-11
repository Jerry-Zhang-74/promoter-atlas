# v5.3.0-mpra-scratch · K562 MPRA 随机初始化对照 / Random-initialization control for K562 MPRA

## 中文

从随机初始化训练同类卷积网络，提供 MPRA 微调和冻结迁移两种方案的对照。输入为 GSE118242 K562 实验的 164 bp 合成核心启动子，输出 RNA/DNA 比值的 Mean_log10 活性。

三个版本的差别是编码器初始化与更新方式。比较时同时查看测试序列相关性和家族内变体排序；对应指标保存在本目录 evaluation 中。

### 查看与复现

- [模型卡](cards/mpra_scratch_card.txt)
- [评估与诊断](evaluation/)
- [训练日志](logs/)
- [三种 MPRA 训练模式](../../../scripts/research/training/train_mpra_finetune.py)
- [下载此版权重与档案](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.3.0-mpra-scratch)

原实验名称：`mpra_scratch`。权重源文件时间：`2026-09-10T17:25:08.151474+08:00`。历史流程的启动方法见[复现指南](../../../docs/REPRODUCIBILITY.md#4-历史流程与原始数据)。

## English

This trains the same convolutional architecture from random initialization, providing the control for the fine-tuned and frozen transfer variants. Inputs are 164 bp synthetic core promoters from the GSE118242 K562 assay; the output is Mean_log10 activity of the RNA/DNA ratio.

The three variants differ in encoder initialization and updating. Compare both test-sequence correlation and within-family variant ranking using the metrics in this directory’s evaluation folder.

### Inspect and reproduce

- [Model card](cards/mpra_scratch_card.txt)
- [Evaluation and diagnostics](evaluation/)
- [Training logs](logs/)
- [Three MPRA training modes](../../../scripts/research/training/train_mpra_finetune.py)
- [Download weights and archive](https://github.com/Jerry-Zhang-74/promoter-atlas/releases/tag/v5.3.0-mpra-scratch)

Original experiment name: `mpra_scratch`. Source checkpoint timestamp: `2026-09-10T17:25:08.151474+08:00`. See the [reproduction guide](../../../docs/REPRODUCIBILITY.md#english) for running historical workflows.
