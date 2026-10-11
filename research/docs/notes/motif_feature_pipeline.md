# Motif feature pipeline v1

## Purpose

This document freezes the first formal motif feature schema for promoter_atlas.

The current step is schema freeze only.
PWM source freeze and motif scanning are still pending.

## Current project state

- Sequence memmap for core promoters already exists.
- `features/motif/` was empty before this schema step.
- Phase 6b now freezes:
  - motif panel
  - motif pair whitelist
  - motif feature manifest

## Scope of v1 motif schema

### Training-aligned scope

The first motif feature version is aligned to the current core promoter window:

- promoter window: up200
- primary key: `record_id`

### Formal outputs planned for later build steps

- `features/motif/promoter_motif_hits.tsv`
- `features/motif/promoter_motif_agg.tsv`
- `features/motif/promoter_motif_grammar.tsv`

These files are not built yet in this step.

## Frozen panel summary

- motif count: 18
- pair count: 10
- planned feature count: 256

### Tier interpretation

- `A_core`: high-priority promoter-relevant motifs
- `B_support`: useful secondary motifs for expression breadth / stability contrast
- `C_control`: control motifs, mainly for contrast and QC

## Planned single-motif features

For each motif, v1 reserves these promoter-level aggregated features:

- `count_loose`
- `count_strict`
- `presence_strict`
- `density_loose`
- `best_score_norm`
- `best_pos_rel_tss`

Additionally, v1 reserves zone-specific counts for:

- distal: `[-200,-101]`
- mid: `[-100,-41]`
- proximal: `[-40,-1]`

## Planned pair grammar features

For each whitelisted pair, v1 reserves:

- `pair_present`
- `min_spacing_bp`
- `spacing_bin`
- `same_orientation_flag`

## Pending items before real motif scanning

1. Freeze PWM source and accession for every motif in `motif_panel_v1.tsv`
2. Freeze score normalization rule
3. Freeze loose / strict threshold policy
4. Implement scan script against `promoters/memmap` or the aligned FASTA source
5. Build hits / agg / grammar tables
6. Train `train_promoter_model_v2.py`

## Design constraints

This schema is designed to support:

- expression strength prediction
- cross-tissue stability prediction
- motif overlap interpretation
- motif grammar interpretation
- small-step promoter editing for wetlab follow-up

## Important project rule

This motif layer is not for benchmark decoration.
It is a formal interface between promoter sequence, model explanation, and experimental design.
