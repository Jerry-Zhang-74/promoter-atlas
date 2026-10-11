# batch21 · K562 片段序列模型的分组测试 / Grouped testing of K562 fragment models

[中文](#中文) · [English](#english)

## 中文

使用 GSE118242 K562 启动子片段 MPRA，按家族与近同源关系分组，比较 A/B 两臂在新序列上的预测。两臂各运行两个固定种子，测试同时报告 Spearman、MAE、前 10% 富集和组 bootstrap。

`public_source_verification.json` 核实数据来源，`results.json` 保存逐臂逐种子指标和预先固定的选择条件。源码包含分组构建、训练、FASTA 输入验证与同来源序列标识。

- [分组测试结果](reports/results.json)
- [两臂训练实现](../shared/runs/code/batch21/b21_train.py)

---

## English

Use GSE118242 K562 promoter-fragment MPRA with family/near-homology grouping to compare arms A/B on new sequences. Each arm uses two fixed seeds; testing reports Spearman, MAE, top-decile enrichment, and grouped bootstrap.

`public_source_verification.json` verifies the source; `results.json` stores per-arm/per-seed metrics and fixed selection conditions. Source modules cover grouping, training, FASTA validation, and source-membership annotation.

- [Grouped test results](reports/results.json)
- [Two-arm trainer](../shared/runs/code/batch21/b21_train.py)
