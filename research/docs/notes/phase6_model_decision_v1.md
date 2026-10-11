# Phase 6 model decision v1

## Purpose

Freeze the current model decision after the Phase 6 pilot series.

## Current official baseline

- model: promoter_model_v1
- route: sequence_baseline
- checkpoint: models/checkpoints/promoter_model_v1.pt
- Score_E_r2: 0.2367
- Score_S_r2: 0.0642
- Score_B_r2: 0.2268
- predEScorr: NA

## Current best architecture-route pilot

- model: promoter_model_v4_arch_only_pilot
- checkpoint: models/checkpoints/promoter_model_v4_arch_only_pilot.pt
- Score_E_r2: 0.2297
- Score_S_r2: 0.0791
- Score_B_r2: 0.2114
- predEScorr: 0.3494

## Architecture reduction reference

- model: promoter_model_v4_arch2_reduced_pilot
- checkpoint: models/checkpoints/promoter_model_v4_arch2_reduced_pilot.pt
- Score_E_r2: 0.2257
- Score_S_r2: 0.0577
- Score_B_r2: 0.2062
- predEScorr: 0.3751

## Current scientific judgment

- single motif aggregation route did not improve the main objective
- pair-lite grammar route did not improve the main objective
- promoter architecture features carry real independent signal for Score_S
- current best Score_S among pilot models comes from the architecture route
- future upgrades should prioritize stronger sequence/structure modeling rather than adding weak motif summaries

## Versioning rule

- major version increases only when the model architecture family changes
- route-internal iterations remain under the same major version
- architecture route remains under v4
