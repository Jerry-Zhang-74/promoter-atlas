# Data Inventory

## Root Check

- project_root: `promoter_atlas`
- legacy_data_exists: `yes`

## Top-level Entries in legacy_data

- [file] `legacy_data/GRCh38.primary_assembly.genome.fa`
- [file] `legacy_data/GRCh38.primary_assembly.genome.fa.fai`
- [file] `legacy_data/GRCh38.primary_assembly.genome.fa:Zone.Identifier`
- [file] `legacy_data/MD5SUMS`
- [file] `legacy_data/gencode.v49.annotation.gtf`

## Summary by Category

### expression

- file_count: `0`
- total_size: `0.0 B`
- examples: none

### promoter_bed

- file_count: `0`
- total_size: `0.0 B`
- examples: none

### promoter_fasta

- file_count: `1`
- total_size: `2.9 GB`
- examples:
  - `legacy_data/GRCh38.primary_assembly.genome.fa` (2.9 GB)

### feature

- file_count: `0`
- total_size: `0.0 B`
- examples: none

### array_like

- file_count: `0`
- total_size: `0.0 B`
- examples: none

### other

- file_count: `4`
- total_size: `3.1 GB`
- examples:
  - `legacy_data/GRCh38.primary_assembly.genome.fa.fai` (6.3 KB)
  - `legacy_data/GRCh38.primary_assembly.genome.fa:Zone.Identifier` (25.0 B)
  - `legacy_data/MD5SUMS` (3.0 KB)
  - `legacy_data/gencode.v49.annotation.gtf` (3.1 GB)

## Candidate Notes

### Expression
- expected_use: Score_E / Score_S / Score_B
- need_manual_check: matrix shape, unit, gene x tissue or gene x sample

### Promoters
- expected_use: promoter coordinate system and sequence extraction baseline
- need_manual_check: strand-aware, promoter length, TSS definition

### Features
- expected_use: motif / sequence / metadata features
- need_manual_check: row alignment with promoter index

## Next Action

- open likely expression files
- open likely promoter BED / FASTA
- confirm whether old labels are mixed into feature assets
- do not reuse old Score_S or old composite HKG labels directly
