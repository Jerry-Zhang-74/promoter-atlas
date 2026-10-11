# batch53 · R0、Nabs 与 Njoint 训练核心 / Training core for R0, Nabs, and Njoint

[中文](#中文) · [English](#english)

## 中文

实现 batch52 的三个候选：342 维序列统计 Ridge、掩码 CNN 绝对活性模型，以及增加配对差值损失的 CNN。神经模型共享初始化、数据、标准化与训练上限，DEV 按组平均配对 MAE 选 checkpoint。

核心模块覆盖数据适配、损失、GPU 锁、执行台账、原子 checkpoint 和指标回放。源码及详细输入/训练协议位于 shared 中的 batch53_model_core；本批 code_manifest 保存所用实现的哈希。

- [训练核心说明](../shared/runs/code/batch53_model_core/README.md)
- [源码校验清单](reports/code_manifest.json)

---

## English

Implement batch52's three candidates: a 342-feature sequence-statistic Ridge, a masked CNN for absolute activity, and a CNN with additional pairwise-difference loss. Neural models share initialization, data, normalization, and epoch limits, selecting checkpoints by DEV group-mean pair MAE.

Modules cover data adaptation, losses, GPU ownership, execution accounting, atomic checkpoints, and evaluation replay. Source and detailed contracts are in shared/batch53_model_core; this batch's code_manifest retains implementation hashes.

- [Training-core specification](../shared/runs/code/batch53_model_core/README.md)
- [Source checksums](reports/code_manifest.json)
