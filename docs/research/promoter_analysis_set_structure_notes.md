# 分析集结构检查 · Analysis-set structure checks

[中文](#中文) · [English](#english)

## 中文

审核 `labels/build/promoter_analysis_set.tsv` 的列数、二进制标记和数据格式，输出：

- `labels/qc/promoter_analysis_set_structure_audit.tsv`
- `labels/qc/promoter_analysis_set_bad_rows.tsv`
- `labels/qc/promoter_analysis_set_column_null_audit.tsv`

修复结构后再继续 Candidate C 稳定性标签构建。

---

## English

Audit `labels/build/promoter_analysis_set.tsv` for field counts, binary flags, and data formats. Outputs:

- `labels/qc/promoter_analysis_set_structure_audit.tsv`
- `labels/qc/promoter_analysis_set_bad_rows.tsv`
- `labels/qc/promoter_analysis_set_column_null_audit.tsv`

Repair structural issues before proceeding to Candidate C stability labels.
