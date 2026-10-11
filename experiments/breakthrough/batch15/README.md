# batch15 · 反向互补检索与跨折相似性审计 / Reverse-complement retrieval and cross-fold similarity

[中文](#中文) · [English](#english)

## 中文

重新检查启动子检索的反向互补匹配，将确认的启动子相似关系映射到基因组件和原有交叉验证折。审计记录 12,996 条启动子边、1,039 条基因边，其中 795 条基因边跨越原折。

这些关系用于解释此前稳定度特征比较的划分条件，并形成 `metrics_v2` 阶段指标快照。入口为 `audit_metrics.json`，其中保留边类型、数量和具体基因对。

- [跨折相似关系](reports/audit_metrics.json)
- [修订指标快照](reports/metrics_v2/source_metrics.json)

---

## English

Audit reverse-complement promoter matches and map verified similarities to gene components and the existing cross-validation folds. The audit records 12,996 promoter edges and 1,039 gene edges, including 795 gene edges crossing the original folds.

These relationships describe the split conditions of the preceding stability-feature comparisons and inform the `metrics_v2` snapshot. `audit_metrics.json` retains edge types, counts, and gene pairs.

- [Cross-fold similarities](reports/audit_metrics.json)
- [Revised metric snapshot](reports/metrics_v2/source_metrics.json)
