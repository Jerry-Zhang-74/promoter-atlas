# Reference Asset Status

## Available Base Assets

- gtf: `yes`
- genome_fasta: `yes`

## Promoter Definition

- promoter window strategy: upstream only
- promoter windows: `up200`, `up2000`
- coordinate system: BED 0-based half-open
- strand-aware: yes

## Annotation Parse Summary

- gene_records_in_gtf: `78691`
- transcript_records_in_gtf: `507365`

## Generated Files

- transcript_up200: `promoters/bed/gencode_v49_transcript_promoters_up200.bed`
- gene_merged_up200: `promoters/bed/gencode_v49_gene_promoters_merged_up200.bed`
- transcript_up2000: `promoters/bed/gencode_v49_transcript_promoters_up2000.bed`
- gene_merged_up2000: `promoters/bed/gencode_v49_gene_promoters_merged_up2000.bed`
- qc_summary: `promoters/qc/gencode_v49_promoter_summary.tsv`

## Missing Assets For Next Stage

- GTEx expression matrix
- promoter-aligned feature tables
- motif scan results
- old memmap assets

## Next Step

- extract FASTA for up200 and up2000 from genome
- build promoter index tables
- compute sequence QC and basic GC features
- do not import old Score_S or old composite labels
