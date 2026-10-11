# Candidate F Label Specification Draft

## 1. Purpose

Candidate F is an incremental extension on top of the frozen Candidate E label system.

It does **not** replace Candidate E.
It adds a fourth axis, **D_time**, to better approximate long-term promoter durability while preserving the existing formal semantics of:

- **E**: expression strength
- **S_tissue**: cross-tissue stability
- **B**: expression breadth

The goal of Candidate F is to move the project closer to an experimentally testable continuous promoter design system, rather than a benchmark-oriented label expansion.

---

## 2. Scope Boundary

### 2.1 What remains frozen from Candidate E

The following Candidate E semantics remain unchanged:

- `Score_E` remains the rank-scaled form of `E_raw`
- `Score_B` remains the rank-scaled form of `B_raw`
- `Score_S` remains the rank-scaled form of `S_tissue_raw`
- if `S_tissue_raw` is undefined, `Score_S` must stay missing
- if `Score_S` is missing, Candidate E style HKG score must stay missing
- undefined `S_tissue` must **not** be written as zero

### 2.2 What Candidate F adds

Candidate F introduces:

- `D_c_raw`: context-specific temporal durability
- `D_time_raw`: aggregated temporal durability across contexts
- `R_c`: context-specific retention ratio
- `R_time`: aggregated retention statistic across contexts

### 2.3 What Candidate F does not claim

Candidate F does **not** claim that:

- `D_time` is a universal gold-standard label
- FANTOM5 can replace GTEx
- cross-tissue stability and temporal durability are interchangeable
- short-term perturbation response automatically measures long-term durability

---

## 3. Core Semantic Separation

The following distinction is mandatory and must be preserved across all future tables, models, plots, and reports:

- `S_tissue != D_time`

### 3.1 Definition of E

`E` describes promoter-associated expression strength through the current Candidate E gene-derived label system.

### 3.2 Definition of S_tissue

`S_tissue` describes stability across tissues at a snapshot level.
It reflects cross-tissue dispersion among detectable tissues.
It is **not** a time-axis property.

### 3.3 Definition of B

`B` describes detectable breadth across tissues.
It remains important for filtering or stratification, but should not become the dominant experimental design axis.

### 3.4 Definition of D_time

`D_time` describes whether promoter activity remains relatively durable across time within valid biological contexts.
It is intended to reduce selection of hit-and-run style promoters that activate early but fail to sustain expression.

### 3.5 Definition of R_time

`R_time` is a retention-style auxiliary metric intended to distinguish sustained activity from rapid early decay.
It supports interpretation and ranking diagnostics, but does not replace `D_time`.

---

## 4. Formal Label Definitions

### 4.1 Existing Candidate E axes

#### Expression strength

For gene `g`:

`E_raw(g) = (1 / T_g) * sum_t log2(1 + x_(g,t))`

where:

- `x_(g,t)` is mean TPM in tissue `t`
- `T_g` is the number of tissues available for gene `g`

#### Breadth

`B_raw(g) = (1 / T_g) * sum_t I(x_(g,t) >= 1)`

with fixed detectable threshold:

- `tau = 1 TPM`

#### Tissue stability

Define the detectable tissue set:

`D(g) = { t : x_(g,t) >= 1 }`

Only compute `S_tissue_raw` when:

- `B_raw(g) >= 0.30`
- detectable tissue count >= 3

For detectable tissues:

`z_(g,t) = log2(1 + x_(g,t))`

Then:

`S_tissue_raw(g) = - MAD({ z_(g,t) : t in D(g) })`

### 4.2 Candidate F temporal durability

For promoter `p` in biological context `c`, with ordered timepoints `Tau_c`:

`D_c_raw(p) = - MAD({ log2(1 + y_(p,c,tau)) : tau in Tau_c })`

where:

- `y_(p,c,tau)` is promoter activity at timepoint `tau`
- promoter activity should be measured or proxied in a promoter-native assay whenever possible
- context `c` may represent differentiation, developmental progression, stimulation, recovery, or another interpretable time-course setting

### 4.3 Aggregated temporal durability

For promoter `p`, let `C_p` be the set of valid contexts with sufficient time-course observations.
Then:

`D_time_raw(p) = median_(c in C_p) D_c_raw(p)`

### 4.4 Context-specific retention

For valid context `c`:

`R_c(p) = ( y_(p,c,tau_late) + epsilon ) / ( y_(p,c,tau_early) + epsilon )`

where:

- `tau_early` is an early interpretable timepoint
- `tau_late` is a later interpretable timepoint
- `epsilon` is a small positive constant for numerical stability

### 4.5 Aggregated retention

`R_time(p) = median_(c in C_p) R_c(p)`

---

## 5. Data Admissibility Rules for D_time

A context is eligible for `D_c_raw` only if all of the following hold:

1. The context has a clear biological meaning.
2. The timepoints are explicitly ordered.
3. The measurement scale is consistent within the context.
4. The promoter or TSS mapping is explicit enough to support promoter-level interpretation.
5. The number of usable timepoints is sufficient for dispersion estimation.
6. Replicate handling is documented.
7. The context is not obviously dominated by an acute one-off response with no interpretable late phase.

If these conditions are not met, `D_c_raw` must remain missing.

---

## 6. Missingness Semantics

Candidate F must preserve explicit missingness instead of coercing undefined values to zero.

### 6.1 D_time missingness

`D_time_raw` must remain missing when:

- no valid time-course context exists
- promoter/TSS mapping is unresolved
- all candidate contexts fail admissibility checks
- timepoint structure is too sparse or biologically uninterpretable

### 6.2 Score_D missingness

If `D_time_raw` is missing, `Score_D` must remain missing.
It must not be imputed as zero.

### 6.3 R_time missingness

If no valid early/late pairing exists, `R_time` must remain missing.

---

## 7. Normalization Rules

All raw axes in the formal label space should use rank scaling to `[0, 1]` within their defined domain.

For any raw label `Q_raw`:

`Score_Q = rank(Q_raw) / (n - 1)`

This applies to:

- `Score_E`
- `Score_S`
- `Score_B`
- future `Score_D`

Domain restrictions must be respected.
Undefined raw values must remain missing after normalization.

---

## 8. Ranking Strategy

### 8.1 Transitional three-axis ranking

Before `D_time` is formally available at scale, the preferred transitional ranking is:

`Score_HKG_3 = exp( 0.40 * log(Score_E + epsilon) + 0.35 * log(Score_S_tissue + epsilon) + 0.25 * log(Score_B + epsilon) )`

This is preferred over a simple equal-weight geometric mean because it better reflects experimental priorities.

### 8.2 Future four-axis ranking

Once `Score_D` becomes available with acceptable coverage and quality:

`Score_HKG_star = exp( 0.25 * log(Score_E + epsilon) + 0.25 * log(Score_S_tissue + epsilon) + 0.15 * log(Score_B + epsilon) + 0.35 * log(Score_D + epsilon) )`

### 8.3 Interpretation of ranking design

This ranking policy is intended to enforce the following:

- no single axis should dominate by itself
- high expression cannot compensate for poor durability
- breadth matters, but should not dominate the experimental objective
- temporal durability becomes a priority once measurable

---

## 9. Recommended Data Flow

Candidate F must be introduced as an additive side branch, not as a destructive rewrite of Candidate E.

### 9.1 Frozen Candidate E branch

`GTEx expression`
→ `gene_expression_table.tsv`
→ `gene_expression_label_table.tsv`
→ `promoter_label_table.tsv`
→ `promoter_model_input_core.tsv`
→ `promoter_model_input_extended.tsv`

### 9.2 Candidate F temporal branch

`FANTOM5 and other temporal proxy sources`
→ `temporal context inventory`
→ `promoter_temporal_mapping_table.tsv`
→ `context_level_durability_table.tsv`
→ `promoter_durability_table.tsv`
→ left join onto promoter label tables by stable promoter identifier

This branch must not alter frozen Candidate E semantics.

---

## 10. FANTOM5 Positioning

FANTOM5 should be treated as one of the highest-priority candidate data sources for Candidate F because:

- it is based on CAGE
- it measures promoter/TSS-associated activity more directly than bulk gene abundance
- it is better aligned with promoter-sequence-driven modeling

However, FANTOM5 must not be over-claimed.

### 10.1 What FANTOM5 is useful for

- promoter-native activity proxy
- context-specific durability estimation
- promoter/TSS-level temporal signal extraction

### 10.2 What FANTOM5 is not

- not a GTEx replacement
- not a universal whole-genome time-course matrix
- not a single unified gold-standard source for long-term expression durability

---

## 11. Experimental Design Implication

The primary design space remains:

- `E x S_tissue`

`B` should mainly be used as a gate or stratification label.
`D_time` should become an additional decision axis once coverage and label validity are acceptable.

This means Candidate F supports experimental design in two stages:

1. preserve the current `E x S_tissue` continuous design backbone
2. add `D_time` and `R_time` to reduce hit-and-run candidates before wet-lab library nomination

---

## 12. Minimal Output Tables to Define Later

The following Candidate F outputs are expected, but should be generated only after data inventory and mapping rules are frozen:

- `labels/build/promoter_temporal_mapping_table.tsv`
- `labels/build/context_level_durability_table.tsv`
- `labels/build/promoter_durability_table.tsv`
- `labels/qc/candidate_f_durability_missingness_summary.tsv`
- `labels/qc/candidate_f_context_coverage_summary.tsv`
- `docs/notes/candidate_f_temporal_data_inventory.md`

---

## 13. Non-Negotiable Rules

1. Do not rewrite undefined labels as zero.
2. Do not collapse `S_tissue` and `D_time` into one score.
3. Do not treat short-term acute response as automatic durability evidence.
4. Do not overwrite frozen Candidate E outputs.
5. Do not claim universal durability when only context-specific proxy evidence exists.
6. Do not let model convenience override experimental interpretability.

---

## 14. Immediate Next Work After This Spec

After this draft is reviewed, the next concrete step should be:

1. build a FANTOM5 resource inventory
2. document promoter/TSS mapping strategy
3. define context admissibility filters for `D_c_raw`
4. define early/late pairing rules for `R_c`
5. only then design Candidate F build scripts

---

## 15. Draft Status

Status: draft v0.1
Position: specification draft for review before data inventory and build implementation
Compatibility: fully additive relative to frozen Candidate E

