# Promoter Analysis Set Structure Notes

## Purpose

- audit the structural integrity of labels/build/promoter_analysis_set.tsv before rebuilding labels
- verify constant column count across rows
- detect malformed binary flags and basic field-format issues
- identify suspicious rows before the next full label rebuild

## Output Files

- audit summary: `labels/qc/promoter_analysis_set_structure_audit.tsv`
- bad row examples: `labels/qc/promoter_analysis_set_bad_rows.tsv`
- column null audit: `labels/qc/promoter_analysis_set_column_null_audit.tsv`

## Next Step

- if structural audit is clean, rebuild labels using S candidate C
- if structural audit is not clean, repair promoter_analysis_set.tsv first
