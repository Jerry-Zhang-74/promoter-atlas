# Observed labels vs model predictions

This release ships **model predictions only**. It does NOT ship or fabricate observed labels.

Catalog **observed** human gene-level labels are maintained separately and traceably at:
* `labels/build/gene_expression_label_table.tsv`
  (columns include E_raw, S_raw, B_raw, Score_E, Score_S, Score_B, B_min_used)
* `labels/build/gene_expression_table.tsv` (tissue-level mean TPM, the source for the above)
* label trace report: `reports/label_trace_20261001/` (e.g. `gene_labels_all_database.tsv`,
  `label_recompute_audit.tsv`, `labels_explorer_validation.json`)

Definitions used by this model:
* E = `Score_E` (rank-scaled expression strength, [0,1]).
* S = `S_raw` = `-MAD(log2(1+detectable TPM))` over detectable tissues; defined for genes with
  `B_raw >= 0.30` and `>= 3` detectable tissues.
* B = `B_raw` = `detectable_tissue_count / tissue_count`.

A value returned by `promoter_atlas_predict` is a **model prediction** (`is_model_prediction: true`,
`is_catalog_observed: false`). Observed labels must be read from the files above and must never be
reported as predictions, nor predictions as observations.
