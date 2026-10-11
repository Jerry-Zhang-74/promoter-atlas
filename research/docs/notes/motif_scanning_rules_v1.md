# Motif scanning rules v1

## Purpose

This document freezes the first motif PWM sourcing and scanning rules for promoter_atlas.

## External PWM source

- source database: JASPAR
- release used in this step: 2026 CORE vertebrates non-redundant
- matrix format: JASPAR batch text
- local raw file:
  - `data/external/jaspar/JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt`

## Matching policy

`motif_panel_v1.tsv` contains project-level motif identifiers.
These identifiers are not always one-to-one with a unique matrix in JASPAR.

Therefore v1 uses a conservative manifest build policy:

- exact unique TF-name match:
  - auto-assign matrix ID
- exact ambiguous TF-name match:
  - keep unresolved
- no exact match:
  - keep unresolved

Unresolved rows are marked as:

- `pending_manual_review`

## Threshold policy

For v1 motif scanning, both loose and strict thresholds are reserved:

- loose threshold:
  - type: relative_score
  - value: 0.80
- strict threshold:
  - type: relative_score
  - value: 0.90

These are initial defaults and may be revised after pilot scan QC.

## Score normalization

Planned normalized score rule:

- observed_score / theoretical_max_score

## Strand policy

- scan forward strand
- scan reverse complement strand
- keep best hit and count features after strand-aware scanning

## Current scope

This step does not scan promoters yet.
It only freezes the PWM source manifest and scanning defaults.

## Planned next outputs

After manual review and matrix freeze, later steps will build:

- `features/motif/promoter_motif_hits.tsv`
- `features/motif/promoter_motif_agg.tsv`
- `features/motif/promoter_motif_grammar.tsv`
