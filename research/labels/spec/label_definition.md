# Label Definition

## Project Goal

- explain real promoter expression rules
- support experimentally testable continuous promoter library design
- avoid semantic collapse between expression and stability
- keep modeling outputs interpretable and auditable

## Core Principle

- do not use one composite score as the only label
- separate expression strength, cross-tissue stability, and expression breadth
- keep HKG preference as a downstream selection aid, not as the main training target

## Primary Outputs

### Score_E
- meaning: expression strength
- target type: continuous
- biological intent: how strongly the promoter drives expression
- expected source: GTEx-derived gene-level expression aggregation
- should not include cross-tissue uniformity inside its formula

### Score_S
- meaning: pure cross-tissue stability
- target type: continuous
- biological intent: how invariant expression is across tissues after separating from raw expression magnitude
- must not reuse the old formulation that effectively gated or regressed stability on expression strength
- must remain meaningful for both high-expression and moderate-expression promoters

### Score_B
- meaning: expression breadth
- target type: continuous or bounded fraction
- biological intent: in how many tissues the promoter-associated gene is detectably expressed
- recommended role: entry gate or stratification variable, not the main experimental axis

### Score_HKG
- meaning: housekeeping-like preference score
- target type: continuous ranking aid
- biological intent: summarize suitability for HKG-like design without pretending to be the same as stability
- recommended role: candidate ranking and wet-lab prioritization
- must be downstream of E / S / B, not a replacement for them

## Experimental Design Space

- primary design axes: E × S
- B is not a main experimental axis
- HKG preference is not a model main output axis

## Modeling Recommendation

- build promoter-only baseline with multi-output prediction for E / S / B
- evaluate up200 as the first clean baseline
- compare up2000 only under the same QC gate
- keep uncertainty and QC tags attached to every promoter record

## Sequence Input Recommendation

- core training baseline should start from labels/build/promoter_analysis_set.tsv where core_model_pass = 1
- extended context analysis can use extended_context_pass = 1 as a paired condition
- qc_fail records should not enter the first baseline model

## Prohibited Old Logic

- do not reuse old Score_S if it mixes expression gate with stability
- do not explicitly fit S as a deterministic function of E
- do not collapse E / S / B into a single benchmark-style score
- do not use AUC / R2 / Spearman as the project objective
