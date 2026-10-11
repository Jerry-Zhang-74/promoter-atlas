# batch45 · 合成启动子优化轨迹与实测读出 / Synthetic-promoter optimization trajectories and measurements

[中文](#中文) · [English](#english)

## 中文

整理作者合成启动子文库中的 3,556 个 barcode 记录，聚合为 889 条独特 DNA：455 条 70 bp 短片段、434 条 232 bp 长片段。每个 RNA 重复先在同一 DNA 的有限 barcode 值上求均值，再汇总两次重复。

沿优化代次比较 PARM 预测分数变化和 K562/HepG2 实测变化，分别保留各自单位。`metric_definitions_denominators.json` 明确片段、barcode、重复和搜索谱系的分母，类别汇总见 `table_category_summary.json`。

- [聚合与分母定义](reports/metric_definitions_denominators.json)
- [文库类别汇总](reports/table_category_summary.json)

---

## English

Organize 3,556 barcode records from the authors' synthetic-promoter library into 889 distinct DNA sequences: 455 short 70 bp fragments and 434 long 232 bp fragments. Within each RNA replicate, average finite barcode values for a DNA sequence, then average the two replicates.

Compare changes in PARM scores and measured K562/HepG2 activity across optimization generations, retaining their respective units. `metric_definitions_denominators.json` defines fragment, barcode, replicate, and search-lineage denominators; `table_category_summary.json` summarizes categories.

- [Aggregation and denominators](reports/metric_definitions_denominators.json)
- [Library-category summary](reports/table_category_summary.json)
