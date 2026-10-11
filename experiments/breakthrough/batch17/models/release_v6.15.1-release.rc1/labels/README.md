# 观察标签与模型预测 · Observed labels and predictions

[中文](#中文) · [English](#english)

## 中文

推理接口返回模型预测，标记为 `is_model_prediction: true`、`is_catalog_observed: false`。

观察标签来源为 `labels/build/gene_expression_label_table.tsv`（E_raw、S_raw、B_raw、Score_E、Score_S、Score_B、B_min_used），其组织均值 TPM 来源为 `labels/build/gene_expression_table.tsv`。追溯记录位于 `reports/label_trace_20261001/`，包括 `gene_labels_all_database.tsv`、`label_recompute_audit.tsv` 和 `labels_explorer_validation.json`。

本模型 E 使用 [0,1] 的 Score_E；S 使用可检出组织 `-MAD(log2(1+TPM))`，定义条件为 B_raw≥0.30 且至少 3 个可检出组织；B 使用可检出组织数/总组织数。使用时保留每个值的观察或预测来源。

---

## English

The inference interface returns model predictions with `is_model_prediction: true` and `is_catalog_observed: false`.

Observed labels are in `labels/build/gene_expression_label_table.tsv` (E_raw, S_raw, B_raw, Score_E, Score_S, Score_B, B_min_used), derived from tissue-mean TPM in `labels/build/gene_expression_table.tsv`. Trace records under `reports/label_trace_20261001/` include `gene_labels_all_database.tsv`, `label_recompute_audit.tsv`, and `labels_explorer_validation.json`.

This model uses Score_E in [0,1]; S is detectable-tissue `-MAD(log2(1+TPM))`, defined when B_raw≥0.30 and at least three tissues are detectable; B is detectable tissue count divided by total tissue count. Retain the observed or predicted provenance of each value.
