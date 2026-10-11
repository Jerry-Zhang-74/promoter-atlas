# Promoter Label Join Audit

## Overall

- promoter_rows: `1014730`
- matched_rows: `872582`
- unmatched_rows: `142148`
- matched_frac: `0.859915`

## Exported Modeling Inputs

- core_model_input: `labels/build/promoter_model_input_core.tsv`
- extended_model_input: `labels/build/promoter_model_input_extended.tsv`
- core_rows: `423473`
- extended_rows: `270123`

## Interpretation

- unmatched promoter rows are expected when promoter annotation gene set is broader than GTEx gene matrix
- this mismatch is especially likely for lncRNA, pseudogene, miRNA, and newly annotated loci
- first baseline should use promoter_model_input_core.tsv
- extended context comparison should use promoter_model_input_extended.tsv
