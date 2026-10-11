"""
Build motif feature schema assets for promoter_atlas.

This script freezes the first formal motif feature schema for Phase 6b.

Outputs
-------
- docs/notes/motif_feature_pipeline.md
- features/motif/motif_panel_v1.tsv
- features/motif/motif_pair_whitelist_v1.tsv
- features/motif/promoter_motif_feature_manifest.tsv

Notes
-----
1. This step freezes schema only.
2. It does not perform PWM scanning yet.
3. All paths are relative to the project root.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(".")

DOC_PATH = ROOT / "docs/notes/motif_feature_pipeline.md"
PANEL_PATH = ROOT / "features/motif/motif_panel_v1.tsv"
PAIR_PATH = ROOT / "features/motif/motif_pair_whitelist_v1.tsv"
MANIFEST_PATH = ROOT / "features/motif/promoter_motif_feature_manifest.tsv"


def build_motif_panel():
    rows = [
        {
            "motif_id": "SP1",
            "tf_family": "SP/KLF",
            "tier": "A_core",
            "design_relevance": "high",
            "scan_group": "core_promoter",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "housekeeping_like_core_promoter",
            "pwm_source_status": "pending_freeze",
            "notes": "GC-rich core promoter anchor",
        },
        {
            "motif_id": "SP3",
            "tf_family": "SP/KLF",
            "tier": "A_core",
            "design_relevance": "high",
            "scan_group": "core_promoter",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "housekeeping_like_core_promoter",
            "pwm_source_status": "pending_freeze",
            "notes": "Complementary to SP1 for broad promoter usage",
        },
        {
            "motif_id": "KLF5",
            "tf_family": "SP/KLF",
            "tier": "B_support",
            "design_relevance": "medium",
            "scan_group": "core_promoter",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "GC_box_related_regulation",
            "pwm_source_status": "pending_freeze",
            "notes": "Representative KLF family motif",
        },
        {
            "motif_id": "NFY",
            "tf_family": "NFY",
            "tier": "A_core",
            "design_relevance": "high",
            "scan_group": "core_promoter",
            "threshold_policy": "loose+strict",
            "preferred_zone": "mid_to_proximal",
            "biological_role": "CCAAT_box_core_promoter",
            "pwm_source_status": "pending_freeze",
            "notes": "Important for broad and stable expression",
        },
        {
            "motif_id": "YY1",
            "tf_family": "YY",
            "tier": "A_core",
            "design_relevance": "high",
            "scan_group": "core_promoter",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "promoter_architecture_and_tss_control",
            "pwm_source_status": "pending_freeze",
            "notes": "Promoter architecture regulator",
        },
        {
            "motif_id": "NRF1",
            "tf_family": "NRF",
            "tier": "A_core",
            "design_relevance": "high",
            "scan_group": "housekeeping_like",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "high_energy_housekeeping_expression",
            "pwm_source_status": "pending_freeze",
            "notes": "Common in constitutive promoters",
        },
        {
            "motif_id": "E2F1",
            "tf_family": "E2F",
            "tier": "B_support",
            "design_relevance": "medium",
            "scan_group": "cell_cycle_related",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "cell_cycle_promoter_control",
            "pwm_source_status": "pending_freeze",
            "notes": "Use as regulated-expression contrast axis",
        },
        {
            "motif_id": "E2F4",
            "tf_family": "E2F",
            "tier": "B_support",
            "design_relevance": "medium",
            "scan_group": "cell_cycle_related",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "cell_cycle_promoter_control",
            "pwm_source_status": "pending_freeze",
            "notes": "Pair with NFY for promoter grammar tests",
        },
        {
            "motif_id": "GABPA",
            "tf_family": "ETS",
            "tier": "A_core",
            "design_relevance": "high",
            "scan_group": "core_promoter",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "broad_promoter_usage",
            "pwm_source_status": "pending_freeze",
            "notes": "ETS-like promoter motif representative",
        },
        {
            "motif_id": "ETS1",
            "tf_family": "ETS",
            "tier": "B_support",
            "design_relevance": "medium",
            "scan_group": "regulated_promoter",
            "threshold_policy": "loose+strict",
            "preferred_zone": "mid_to_proximal",
            "biological_role": "ETS_family_contrast",
            "pwm_source_status": "pending_freeze",
            "notes": "Family contrast to GABPA",
        },
        {
            "motif_id": "ELK1",
            "tf_family": "ETS",
            "tier": "B_support",
            "design_relevance": "medium",
            "scan_group": "regulated_promoter",
            "threshold_policy": "loose+strict",
            "preferred_zone": "mid_to_proximal",
            "biological_role": "ETS_family_contrast",
            "pwm_source_status": "pending_freeze",
            "notes": "MAPK-responsive ETS family representative",
        },
        {
            "motif_id": "CREB1",
            "tf_family": "CREB_ATF",
            "tier": "A_core",
            "design_relevance": "high",
            "scan_group": "core_promoter",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "CRE_core_promoter_signal_integration",
            "pwm_source_status": "pending_freeze",
            "notes": "Signal-responsive but promoter-relevant",
        },
        {
            "motif_id": "ATF1",
            "tf_family": "CREB_ATF",
            "tier": "B_support",
            "design_relevance": "medium",
            "scan_group": "core_promoter",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "CRE_like_signal_integration",
            "pwm_source_status": "pending_freeze",
            "notes": "Support motif for CRE family",
        },
        {
            "motif_id": "TBP",
            "tf_family": "TBP",
            "tier": "A_core",
            "design_relevance": "high",
            "scan_group": "tata_related",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "tata_box_related_tss_control",
            "pwm_source_status": "pending_freeze",
            "notes": "Keep despite expected low prevalence in HKG-like promoters",
        },
        {
            "motif_id": "USF1",
            "tf_family": "USF",
            "tier": "B_support",
            "design_relevance": "medium",
            "scan_group": "broad_expression",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "e_box_related_promoter_regulation",
            "pwm_source_status": "pending_freeze",
            "notes": "Useful for contrast with MAX-like motifs",
        },
        {
            "motif_id": "MAX",
            "tf_family": "bHLH",
            "tier": "B_support",
            "design_relevance": "medium",
            "scan_group": "broad_expression",
            "threshold_policy": "loose+strict",
            "preferred_zone": "proximal",
            "biological_role": "e_box_related_promoter_regulation",
            "pwm_source_status": "pending_freeze",
            "notes": "Generic E-box family representative",
        },
        {
            "motif_id": "RFX1",
            "tf_family": "RFX",
            "tier": "B_support",
            "design_relevance": "medium",
            "scan_group": "broad_expression",
            "threshold_policy": "loose+strict",
            "preferred_zone": "mid_to_proximal",
            "biological_role": "promoter_occupancy_contrast",
            "pwm_source_status": "pending_freeze",
            "notes": "Useful for stable-expression contrast",
        },
        {
            "motif_id": "CTCF",
            "tf_family": "CTCF",
            "tier": "C_control",
            "design_relevance": "low",
            "scan_group": "control",
            "threshold_policy": "loose+strict",
            "preferred_zone": "distal_to_mid",
            "biological_role": "negative_or_structure_control",
            "pwm_source_status": "pending_freeze",
            "notes": "Control motif, not expected as primary core-promoter driver",
        },
    ]
    return pd.DataFrame(rows)


def build_pair_whitelist():
    rows = [
        {"pair_id": "SP1__NFY", "motif_a": "SP1", "motif_b": "NFY", "priority": "high", "rationale": "housekeeping_like_core_pair"},
        {"pair_id": "SP1__YY1", "motif_a": "SP1", "motif_b": "YY1", "priority": "high", "rationale": "core_promoter_architecture_pair"},
        {"pair_id": "SP1__NRF1", "motif_a": "SP1", "motif_b": "NRF1", "priority": "high", "rationale": "broad_expression_pair"},
        {"pair_id": "NFY__YY1", "motif_a": "NFY", "motif_b": "YY1", "priority": "high", "rationale": "stable_promoter_architecture_pair"},
        {"pair_id": "NFY__E2F1", "motif_a": "NFY", "motif_b": "E2F1", "priority": "medium", "rationale": "regulated_vs_broad_expression_contrast"},
        {"pair_id": "NFY__E2F4", "motif_a": "NFY", "motif_b": "E2F4", "priority": "medium", "rationale": "cell_cycle_promoter_grammar"},
        {"pair_id": "GABPA__NFY", "motif_a": "GABPA", "motif_b": "NFY", "priority": "medium", "rationale": "ets_nfy_promoter_pair"},
        {"pair_id": "CREB1__SP1", "motif_a": "CREB1", "motif_b": "SP1", "priority": "medium", "rationale": "signal_integrating_promoter_pair"},
        {"pair_id": "TBP__SP1", "motif_a": "TBP", "motif_b": "SP1", "priority": "medium", "rationale": "tata_vs_gc_core_architecture"},
        {"pair_id": "USF1__MAX", "motif_a": "USF1", "motif_b": "MAX", "priority": "low", "rationale": "ebox_family_control_pair"},
    ]
    return pd.DataFrame(rows)


def build_feature_manifest(panel_df, pair_df):
    rows = []

    zone_defs = [
        ("distal", "[-200,-101]"),
        ("mid", "[-100,-41]"),
        ("proximal", "[-40,-1]"),
    ]

    single_feature_templates = [
        ("count_loose", "numeric", "motif hit count under loose threshold"),
        ("count_strict", "numeric", "motif hit count under strict threshold"),
        ("presence_strict", "binary", "whether strict hit exists"),
        ("density_loose", "numeric", "loose hit count divided by promoter length"),
        ("best_score_norm", "numeric", "best normalized motif score"),
        ("best_pos_rel_tss", "numeric", "best-hit start position relative to TSS"),
    ]

    zone_feature_templates = [
        ("count_loose", "numeric", "zone-specific loose hit count"),
        ("count_strict", "numeric", "zone-specific strict hit count"),
    ]

    for motif_id in panel_df["motif_id"].tolist():
        for suffix, dtype, desc in single_feature_templates:
            rows.append(
                {
                    "feature_name": f"{motif_id}_{suffix}",
                    "feature_level": "single_motif",
                    "feature_type": dtype,
                    "source_object": motif_id,
                    "phase6b_status": "planned",
                    "description": desc,
                }
            )
        for zone_name, zone_range in zone_defs:
            for suffix, dtype, desc in zone_feature_templates:
                rows.append(
                    {
                        "feature_name": f"{motif_id}_{zone_name}_{suffix}",
                        "feature_level": "single_motif_zone",
                        "feature_type": dtype,
                        "source_object": motif_id,
                        "phase6b_status": "planned",
                        "description": f"{desc} in zone {zone_name} {zone_range}",
                    }
                )

    pair_feature_templates = [
        ("pair_present", "binary", "whether both motifs are present"),
        ("min_spacing_bp", "numeric", "minimum spacing between motif pair hits"),
        ("spacing_bin", "categorical", "binned minimum spacing"),
        ("same_orientation_flag", "binary", "whether best pair has same orientation"),
    ]

    for pair_id in pair_df["pair_id"].tolist():
        for suffix, dtype, desc in pair_feature_templates:
            rows.append(
                {
                    "feature_name": f"{pair_id}_{suffix}",
                    "feature_level": "motif_pair",
                    "feature_type": dtype,
                    "source_object": pair_id,
                    "phase6b_status": "planned",
                    "description": desc,
                }
            )

    return pd.DataFrame(rows)


def build_doc_text(panel_df, pair_df, manifest_df):
    text = f"""# Motif feature pipeline v1

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

- motif count: {len(panel_df)}
- pair count: {len(pair_df)}
- planned feature count: {len(manifest_df)}

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
"""
    return text


def main():
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    panel_df = build_motif_panel()
    pair_df = build_pair_whitelist()
    manifest_df = build_feature_manifest(panel_df, pair_df)
    doc_text = build_doc_text(panel_df, pair_df, manifest_df)

    panel_df.to_csv(PANEL_PATH, sep="\t", index=False)
    pair_df.to_csv(PAIR_PATH, sep="\t", index=False)
    manifest_df.to_csv(MANIFEST_PATH, sep="\t", index=False)

    with open(DOC_PATH, "w", encoding="utf-8") as f:
        f.write(doc_text)

    print("motif schema build finished")
    print(f"panel: {PANEL_PATH}")
    print(f"pair whitelist: {PAIR_PATH}")
    print(f"manifest: {MANIFEST_PATH}")
    print(f"doc: {DOC_PATH}")
    print(f"motif_count: {len(panel_df)}")
    print(f"pair_count: {len(pair_df)}")
    print(f"feature_count: {len(manifest_df)}")


if __name__ == "__main__":
    main()
