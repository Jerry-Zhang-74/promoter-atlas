# batch39 · HEK293FT 270 bp 寡核苷酸读出准备 / HEK293FT 270 bp oligo-readout preparation

[中文](#中文) · [English](#english)

## 中文

解析 GSE325670 / MAPT 研究补充表 s5（OligoMPRA Heks），核对标签中的坐标、链方向与设计序列。作者设计使用半开区间，基因组插入为 270 bp；两侧各加 15 bp 后合成为 300 bp。

本批把该片段活性读出与先前的变体/ref 效应分开描述，整理匹配条件、来源表和目标情境。`source_audit_table.tsv` 是人工阅读入口，`finalize_b.py` 保存目标情境定义。

- [来源核查表](reports/source_audit_table.tsv)
- [片段任务定义](code/batch39/finalize_b.py)
- [补充表匹配](code/batch39/parse_s5_join.py)

---

## English

Parse supplementary table s5 (OligoMPRA Heks) from GSE325670 / the MAPT study and reconcile coordinates, strand, and design sequences. The author design uses half-open intervals: a 270 bp genomic insert plus two 15 bp flanks yields a 300 bp oligo.

This batch defines fragment-activity measurements separately from the earlier variant/ref effects, recording joins, source tables, and assay context. Start with `source_audit_table.tsv`; `finalize_b.py` contains the target-context specification.

- [Source-audit table](reports/source_audit_table.tsv)
- [Fragment-task definition](code/batch39/finalize_b.py)
- [Supplementary-table join](code/batch39/parse_s5_join.py)
