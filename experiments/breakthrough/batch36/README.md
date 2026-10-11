# batch36 · MAPT 实测变体效应与序列匹配 / MAPT measured variant effects and sequence matching

[中文](#中文) · [English](#english)

## 中文

读取论文 `media-1.xlsx` 补充表，将 HEK293FT 与神经元的 bcalm logFC 变体效应匹配到作者设计 FASTA。249 bp 合成寡核苷酸由两侧各 15 bp 固定序列和 219 bp 可变插入组成；HEK293FT 饱和变体表的 8,000 行全部匹配。

`cohort_freeze_audit.json` 记录细胞背景、序列拆分、重复/反向互补检查和逐表匹配。`label_source_manifest.json` 追踪每一类标签的实测来源，供后续 ref/alt 预测差值诊断。

- [变体队列匹配审计](reports/cohort_freeze_audit.json)
- [实测标签来源](reports/label_source_manifest.json)
- [队列构建实现](code/batch36/freeze_cohort.py)

---

## English

Read supplementary `media-1.xlsx` tables and join HEK293FT/neuron bcalm logFC variant effects to the authors' design FASTA. Each 249 bp synthesized oligo comprises two fixed 15 bp flanks and a 219 bp variable insert; all 8,000 HEK293FT saturation-variant rows match.

`cohort_freeze_audit.json` records cell context, sequence decomposition, duplicate/reverse-complement checks, and table joins. `label_source_manifest.json` traces measured label sources for subsequent ref/alt prediction-difference diagnostics.

- [Variant-cohort join audit](reports/cohort_freeze_audit.json)
- [Measured-label sources](reports/label_source_manifest.json)
- [Cohort construction](code/batch36/freeze_cohort.py)
