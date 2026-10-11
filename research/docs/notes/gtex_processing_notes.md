# GTEx Processing Notes

## Input

- expression_matrix: `data/raw/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz`
- sample_annotation: `data/raw/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt`

## Output

- gene_expression_table: `labels/build/gene_expression_table.tsv`
- expression_summary: `labels/qc/gtex_expression_summary.tsv`

## Method

- stream one gene row at a time from gzipped GCT
- do not store all genes in memory
- aggregate TPM by tissue
- output mean_tpm and median_tpm per gene x tissue
- remove gene version suffix from gene_id

## Memory Strategy

- keep only sample annotation mapping and tissue column groups in memory
- compute statistics for one gene at a time
- write results directly to disk
