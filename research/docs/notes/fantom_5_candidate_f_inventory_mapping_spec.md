# FANTOM5 Candidate F Inventory and Mapping Specification Draft

## 1. Purpose

This document defines how FANTOM5 should be inventoried and connected to Candidate F without altering the frozen Candidate E branch.

The purpose is not to force an immediate durability label build.
The purpose is to create a controlled intake path for promoter-native temporal proxy data.

---

## 2. Position of FANTOM5 in promoter_atlas

FANTOM5 is treated as a high-priority Candidate F source because it is promoter-centered and CAGE-based.
This makes it more aligned with promoter-sequence-driven modeling than bulk RNA-seq gene abundance resources.

However, FANTOM5 is not a universal long-term durability gold label resource.
It is a heterogeneous collection of promoter/TSS activity measurements across biological states, including selected time-course experiments.

Therefore, in promoter_atlas:

- FANTOM5 is a **temporal proxy source**
- FANTOM5 is a **promoter/TSS activity source**
- FANTOM5 is **not** a direct replacement for GTEx
- FANTOM5 is **not** a single unified global time-axis matrix

---

## 3. What data content from FANTOM5 matters here

Candidate F should prioritize the following FANTOM5 content classes:

1. **Promoter / TSS peak definitions**
   - robust CAGE peaks
   - permissive peaks only for exploration, not default label build

2. **Promoter-level activity summaries**
   - counts
   - TPM
   - max TPM
   - total TPM

3. **Individual sample activity tracks**
   - primary cells
   - tissues
   - cell lines
   - fractionation
   - time course

4. **Time-course sample subsets**
   - only contexts with explicit temporal ordering should enter Candidate F durability review

5. **Metadata required for context grouping**
   - sample identifiers
   - biological context labels
   - timepoint labels
   - assay subtype
   - species
   - genome build
   - replicate annotations if available

---

## 4. Inventory-first principle

No Candidate F durability label should be computed directly from raw FANTOM5 file discovery.

The intake must go through an explicit inventory stage.

### 4.1 Required inventory outputs

Expected inventory-stage outputs:

- `data/external/fantom5/fantom5_resource_inventory.tsv`
- `data/external/fantom5/fantom5_context_inventory.tsv`
- `docs/notes/fantom5_candidate_f_mapping_notes.md`

### 4.2 Resource inventory fields

The resource inventory should include at least:

- `resource_id`
- `source_group`
- `file_name`
- `file_role`
- `species`
- `genome_build`
- `assay_type`
- `signal_unit`
- `sample_category`
- `contains_timecourse`
- `contains_peak_coordinates`
- `contains_expression_values`
- `download_status`
- `parse_status`
- `notes`

### 4.3 Context inventory fields

The context inventory should include at least:

- `context_id`
- `context_name`
- `sample_category`
- `biological_process_type`
- `species`
- `genome_build`
- `timepoint_count`
- `timepoint_labels`
- `time_order_defined`
- `replicate_info_available`
- `signal_unit`
- `promoter_level_ready`
- `candidate_for_durability`
- `exclusion_reason`
- `notes`

---

## 5. Primary mapping objects

Candidate F should distinguish four different mapping objects.

### 5.1 FANTOM peak object

A FANTOM peak object is a CAGE-defined promoter/TSS peak used as the unit of FANTOM promoter-level activity.

Suggested fields:

- `fantom_peak_id`
- `chrom`
- `start`
- `end`
- `strand`
- `peak_class`
- `associated_gene_symbol`
- `associated_gene_id`
- `source_build`

### 5.2 promoter_atlas promoter object

A promoter_atlas promoter object is the stable promoter unit already used in Candidate E tables.

Suggested core key:

- `promoter_id`

### 5.3 peak-to-promoter mapping object

A peak-to-promoter mapping object connects FANTOM CAGE peaks to promoter_atlas promoter intervals.

Suggested fields:

- `fantom_peak_id`
- `promoter_id`
- `overlap_bp`
- `peak_center_to_promoter_anchor_distance`
- `strand_match`
- `mapping_class`
- `mapping_confidence`
- `mapping_rule_version`

### 5.4 context activity object

A context activity object stores promoter activity for a mapped promoter in one context at one timepoint.

Suggested fields:

- `context_id`
- `timepoint_label`
- `time_rank`
- `fantom_peak_id`
- `promoter_id`
- `activity_value`
- `signal_unit`
- `replicate_aggregation_rule`
- `activity_qc_flag`

---

## 6. Mapping policy

### 6.1 Mapping goals

The mapping policy should maximize interpretability and minimize false promoter assignment.

The goal is not maximal recall.
The goal is trustworthy promoter-level temporal evidence.

### 6.2 Default mapping preference

Default priority:

1. strand-consistent direct interval overlap
2. peak center within promoter anchor window
3. nearest compatible promoter only when overlap-based mapping is unavailable and ambiguity is low

### 6.3 Ambiguity handling

A FANTOM peak must not be blindly assigned when:

- it overlaps multiple promoter_atlas promoters with similar support
- strand assignment is inconsistent
- the peak lies in a dense bidirectional region without clean promoter attribution
- genome build reconciliation is unresolved

Such cases should remain ambiguous and be excluded from Candidate F label building unless a later rule resolves them explicitly.

### 6.4 Mapping confidence levels

Suggested confidence levels:

- `high`: unique strand-consistent overlap with strong anchor consistency
- `medium`: unique nearest-compatible mapping with acceptable distance
- `low`: ambiguous or weak-support mapping
- `reject`: mapping unsuitable for Candidate F label generation

Only `high` and selected `medium` mappings should enter durability builds.

---

## 7. Build compatibility and genome coordination

FANTOM5 resources may exist in different genome builds or reprocessed assemblies.

Before durability computation, inventory must explicitly record:

- source genome build
- promoter_atlas build target
- whether liftover or coordinate harmonization is required
- whether peak identifiers remain stable after harmonization

Candidate F must not mix coordinate systems implicitly.

---

## 8. Context admissibility for D_c_raw

Not every FANTOM5 sample set should be treated as a durability context.

A context is admissible for `D_c_raw` only when:

1. It has a biologically coherent label.
2. It contains explicitly ordered timepoints.
3. It captures more than an immediate one-off perturbation snapshot.
4. The promoter activity signal is comparable across the timepoints.
5. The mapped promoters in that context pass mapping confidence filters.
6. Replicate handling is documented.
7. The late phase is biologically interpretable.

Examples of likely admissible contexts:

- differentiation time course
- developmental progression
- activation followed by sustained response or recovery

Examples requiring caution or likely exclusion:

- acute single-pulse response without interpretable late phase
- sparse two-point designs with unclear timing meaning
- contexts where promoter assignment is unstable across nearby TSS clusters

---

## 9. Rules for D_c_raw and R_c computation preparation

This stage is still inventory and mapping design, not score production.

But the resource design must anticipate later computation.

### 9.1 Required prerequisites for D_c_raw

Before `D_c_raw` is computed, the pipeline must know:

- valid `context_id`
- ordered `time_rank`
- promoter-level mapped activity values
- within-context comparable unit
- missingness policy
- replicate aggregation policy

### 9.2 Required prerequisites for R_c

Before `R_c` is computed, the pipeline must define:

- which timepoint is `early`
- which timepoint is `late`
- whether intermediate points are ignored or used for QC
- how to handle very low early values
- which epsilon is used

---

## 10. Recommended staged outputs

The Candidate F FANTOM5 intake should be staged as follows.

### Stage A: inventory

- `data/external/fantom5/fantom5_resource_inventory.tsv`
- `data/external/fantom5/fantom5_context_inventory.tsv`

### Stage B: peak and sample parsing

- `data/interim/fantom5/fantom5_peak_table.tsv`
- `data/interim/fantom5/fantom5_sample_table.tsv`
- `data/interim/fantom5/fantom5_timecourse_sample_table.tsv`

### Stage C: mapping

- `labels/build/fantom_peak_to_promoter_mapping.tsv`
- `labels/qc/fantom_peak_to_promoter_mapping_summary.tsv`

### Stage D: context-ready activity tables

- `labels/build/fantom_context_activity_long.tsv`
- `labels/qc/fantom_context_activity_missingness.tsv`

### Stage E: future durability tables

- `labels/build/context_level_durability_table.tsv`
- `labels/build/promoter_durability_table.tsv`

---

## 11. QC requirements

The inventory and mapping stage must produce QC outputs before any label build is allowed.

Minimum QC summaries:

1. resource parse success by file
2. context counts by sample category
3. count of contexts with valid temporal ordering
4. count of peaks by genome build and peak class
5. fraction of peaks mapped to promoter_atlas promoters
6. mapping confidence distribution
7. ambiguous mapping count
8. contexts excluded from Candidate F durability and reasons

---

## 12. Wet-lab relevance filter

Candidate F should only advance FANTOM-derived durability evidence that can improve downstream experimental nomination.

That means the intake should prioritize contexts and mappings that help answer:

- does this promoter sustain activity rather than spike and collapse?
- is the temporal pattern interpretable enough to guide candidate selection?
- can the inferred durability constraint later be linked back to promoter sequence edits?

If a FANTOM-derived signal does not improve these questions, it should not be promoted into the main Candidate F label layer.

---

## 13. Non-negotiable implementation rules

1. Do not overwrite Candidate E tables.
2. Do not collapse all FANTOM samples into one pseudo-global time axis.
3. Do not treat every FANTOM context as durability-eligible.
4. Do not default ambiguous peak-to-promoter mapping into low-confidence labels.
5. Do not use permissive peaks as default label anchors.
6. Do not claim long-term durability when only short-term response evidence exists.

---

## 14. Immediate next step after this document

The next concrete design document should define:

- context admissibility checklist for Candidate F
- promoter/TSS reconciliation policy in more detail
- early/late pairing rules for retention
- preferred file acquisition order from FANTOM5 resources

---

## 15. Draft Status

Status: draft v0.1
Scope: FANTOM5 intake, inventory, and mapping design for Candidate F
Dependency: compatible with frozen Candidate E branch
