# batch40 · 作者寡核苷酸设计与标签连接 / Author oligo-design and label joins

[中文](#中文) · [English](#english)

## 中文

获取作者的 `GSM9580234_design_rmIllegalChars.fa.gz`，按完整设计标识连接 s5 测量，验证固定两侧序列、插入长度、链方向和标签坐标。记录 7,537 个基因组条目及 85 个对照的精确唯一匹配。

分层抽查覆盖 MAPT 主区域、其他染色体、较大家族、对照和窗口重叠。`source20_check.json` 给出 23 条抽查的覆盖情形；构建及分组逻辑见 finalize/grouping 模块。

- [分层原始数据核对](reports/source20_check.json)
- [来源审计](reports/source_audit_table.tsv)
- [区间分组实现](code/batch40/grouping.py)

---

## English

Retrieve the authors' `GSM9580234_design_rmIllegalChars.fa.gz` and join s5 measurements by complete design identifiers, checking flanks, insert length, strand, and label coordinates. Records describe exact unique joins for 7,537 genomic entries and 85 controls.

Stratified checks cover the main MAPT region, other chromosomes, large families, controls, and overlapping windows. `source20_check.json` describes 23 checked rows; finalize/grouping modules implement construction and grouping.

- [Stratified source checks](reports/source20_check.json)
- [Source audit](reports/source_audit_table.tsv)
- [Interval grouping](code/batch40/grouping.py)
