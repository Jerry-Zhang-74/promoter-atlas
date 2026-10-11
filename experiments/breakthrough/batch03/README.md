# batch03 · 从头训练的基因汇总模型与构建体诊断 / Scratch gene aggregation and construct diagnostics

[中文](#中文) · [English](#english)

## 中文

比较单启动子 CNN、同基因多启动子 mean/max 汇总 CNN，以及序列组成 Ridge，检查冻结表征结果能否通过从头训练重现。报告同时列出完整测试集和历史 donor 未暴露子集；bag 模型两者的 E Spearman 分别为 0.7491 与 0.4527。

另一条分析使用 14 个本地构建体，按构建体留一和家族留一评估读出。`trackB_v63_rerun_results.json` 逐个记录 ACTB、GAPDH、CMV 的真实端点与预测，便于区分基因级排序和构建体活性任务。

- [基因模型比较](reports/trackA_summary.json)
- [构建体逐项复算](reports/trackB_v63_rerun_results.json)

---

## English

Compare a single-promoter CNN, a mean/max gene-bag CNN, and a composition-based Ridge model trained from scratch. Reports separate the full test set from the historical donor-unexposed subset; the bag model's E Spearman is 0.7491 and 0.4527, respectively.

A second analysis evaluates readouts on 14 local constructs using construct-wise and family-wise leave-one-out designs. `trackB_v63_rerun_results.json` records observed endpoints and predictions for ACTB, GAPDH, and CMV constructs, connecting the gene-ranking and construct-activity tasks.

- [Gene-model comparison](reports/trackA_summary.json)
- [Construct-level replay](reports/trackB_v63_rerun_results.json)
