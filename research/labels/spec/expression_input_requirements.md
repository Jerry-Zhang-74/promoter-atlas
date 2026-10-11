# 表达输入要求 · Expression input requirements

[中文](#中文) · [English](#english)

## 中文

### 上游资产

准备 GTEx 表达矩阵、与 GENCODE v49 `gene_id` 兼容的基因映射、组织元数据，以及样本级或组织聚合后的表达表。

最少字段为 `gene_id`、`tissue_id` 或 `tissue_name`、`expression_value`、`expression_unit`。

### 表达与标签语义

使用有生物学解释的归一化表达值，注明矩阵是基因×样本还是基因×组织，保留组织覆盖信息并记录过滤阈值。

- E：记录跨组织或组织内样本的表达强度汇总规则。
- S：衡量跨组织稳定性；所用方差归一化或残差处理需明确记录。
- B：明确逐组织可检出阈值，便于复核。

表达标签首先通过 `gene_id` 连接启动子。首个基线允许同一基因的转录本启动子共享基因级标签，转录本特异性细化可作为后续分支。

---

## English

### Upstream assets

Prepare a GTEx-derived expression matrix, a gene mapping compatible with GENCODE v49 `gene_id`, tissue metadata, and sample-level or tissue-aggregated expression summaries.

Minimum fields: `gene_id`, `tissue_id` or `tissue_name`, `expression_value`, and `expression_unit`.

### Expression and target semantics

Use biologically interpretable normalized values, specify whether the matrix is gene-by-sample or gene-by-tissue, retain tissue coverage, and document filtering thresholds.

- E: document the strength summary across tissues or samples within tissues.
- S: measure cross-tissue stability and document any variance normalization or residualization.
- B: state an auditable tissue-level detection threshold.

Join expression labels to promoters by `gene_id`. The first baseline allows transcript promoters from the same gene to share its gene-level label. Transcript-aware refinements can be developed as a subsequent branch.
