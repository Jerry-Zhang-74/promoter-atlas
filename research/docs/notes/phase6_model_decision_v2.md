# Phase 6 model decision v2

## Purpose

Refresh the Phase 6 model decision after unified evaluation of promoter_model_v1.

## Dual baseline decision

### Official overall baseline

- model: promoter_model_v1
- route: sequence_baseline
- checkpoint: models/checkpoints/promoter_model_v1.pt
- Score_E_r2: 0.2537
- Score_S_r2: 0.0707
- Score_B_r2: 0.2327
- predEScorr: 0.4492

### Official S-focused architecture reference

- model: promoter_model_v4_arch_only_pilot
- route: architecture_route
- checkpoint: models/checkpoints/promoter_model_v4_arch_only_pilot.pt
- Score_E_r2: 0.2297
- Score_S_r2: 0.0791
- Score_B_r2: 0.2114
- predEScorr: 0.3494

## Current scientific judgment

- motif aggregation route is not the main path for Score_S
- pair-lite grammar route is not the main path for Score_S
- promoter architecture carries independent signal for Score_S
- next model upgrades should focus on stronger sequence-structure fusion rather than adding weak motif summaries
