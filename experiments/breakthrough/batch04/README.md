# batch04 · 多启动子采样与确定性池化 / Promoter sampling and deterministic pooling

[中文](#中文) · [English](#english)

## 中文

固定每个基因四个输入槽，比较重复取同一启动子与轮换取不同启动子的 bag CNN，并用三个随机种子检查差异。结果按固定四槽和全部启动子两种推理方式汇总；不同启动子方案的全部启动子 TEST Spearman 均值为 0.7515。

`resume_check.json` 记录恢复检查；`_superseded_nondeterministic_pooling` 保存早期池化实现的结果，当前比较入口是 `consolidated_summary.json`。

- [三种子汇总](reports/consolidated_summary.json)
- [恢复检查](models/resume_check.json)

---

## English

Keep four input slots per gene and compare repeating one promoter against rotating through distinct promoters, using three seeds. Results report both fixed four-slot and all-promoter inference; the diverse-promoter arm has a mean all-promoter TEST Spearman of 0.7515.

`resume_check.json` records recovery checks. `_superseded_nondeterministic_pooling` retains results from the earlier pooling implementation; use `consolidated_summary.json` for the consolidated comparison.

- [Three-seed summary](reports/consolidated_summary.json)
- [Resume checks](models/resume_check.json)
