# batch27 · SuRE 长片段队列准备与 STR 来源补记 / SuRE long-fragment preparation and STR provenance

[中文](#中文) · [English](#english)

## 中文

面向 GSE78709 K562 SuRE 的原生长片段预测，扫描原始计数、设计无标签抽样与同源分组，并核对 hg19 序列获取。数据准备把 iPCR、RNA、质粒计数和组结构分开记录，形成后续目标定义与训练提案。

另对 batch26 补充来源和执行记录，补记文件位于本批目录。`source_exposure_audit.json` 记录公开来源的既有使用情况；实际 SuRE 计数检查延续到 batch28。

- [STR 来源补记](reports/batch26_manifest_addendum.json)
- [来源使用记录](reports/source_exposure_audit.json)
- [SuRE 数据准备](../shared/runs/code/batch27/b27_data_ready.py)

---

## English

Prepare native long-fragment prediction from GSE78709 K562 SuRE by scanning raw counts, designing label-free sampling and homology groups, and checking hg19 sequence retrieval. Data preparation separates iPCR, RNA, plasmid counts, and grouping for later target definition and training design.

This batch also adds provenance and execution details for batch26 in a separate addendum. `source_exposure_audit.json` records prior use of public sources; SuRE count inspection continues in batch28.

- [STR provenance addendum](reports/batch26_manifest_addendum.json)
- [Source-use record](reports/source_exposure_audit.json)
- [SuRE data preparation](../shared/runs/code/batch27/b27_data_ready.py)
