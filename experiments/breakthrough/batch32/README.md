# batch32 · PARM 来源重叠与 293FT 实测端点审计 / PARM source overlap and 293FT endpoint audit

[中文](#中文) · [English](#english)

## 中文

按 hg19 半开区间比较 HEK673 与作者发布的 PARM 文库片段，分列重叠、无重叠和染色体来源缺失的记录。673 条中 613 条与已发布文库区间重叠，59 条无重叠，1 条来源染色体未发布；指标按这些子组重算。

本地部分从 CytoFLEX 测量追踪 GFP 阳性门中位数/同批 WT 阴性中位数的端点，核对 ACTB/GAPDH 的嵌套关系和 CMV 实际序列长度。来源重叠和本地端点分别见 sectionA、sectionB。

- [文库区间重叠](reports/sectionA_metrics.json)
- [本地端点与序列](reports/sectionB_metrics.json)

---

## English

Compare HEK673 against author-distributed PARM library intervals using hg19 half-open coordinates. Of 673 records, 613 overlap distributed intervals, 59 are disjoint, and one lies on an undistributed chromosome. Metrics are recomputed for these subgroups.

The local analysis traces CytoFLEX endpoints defined as GFP-positive-gate median divided by same-batch WT-negative median, checks ACTB/GAPDH sequence nesting, and verifies actual CMV lengths. Section A covers source overlap; section B covers local endpoints.

- [Library-interval overlap](reports/sectionA_metrics.json)
- [Local endpoints and sequences](reports/sectionB_metrics.json)
