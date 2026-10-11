#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the frozen batch11 config.json BEFORE any held-out metric is computed.

Immutable freeze-point for the four predeclared scientific fits (A=Ridge/HGB,
B=Ridge/HGB on continuous raw S). Records source/split/feature/label SHAs,
encoding/window conventions, all constants, transform formulas, model configs
and the four-comparison evaluation convention. Written once before any val/test
number is inspected.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b11_common import (ROOT, EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ,
                        FEATURES_NPZ, MEM400, MEM2000, B10_FEATURES1024,
                        MODEL_OUT, CANDIDATE_VERSION, TRACK, BATCH, SEED,
                        N_BOOTSTRAP, BOOT_LO_PCT, BOOT_HI_PCT, BOOT_TWO_SIDED_PCT,
                        DELTA_MIN, RIDGE, HGB, FEATURE_NAMES_45, FEATURE_NAMES_1024,
                        FEATURE_NAMES_ALL, DETECTABLE_TPM, B_MIN, MIN_DET_TISSUES,
                        N_CANONICAL, M_DEFINED, N_FEATURE_UNIVERSE,
                        N_EXPECTED_ELIGIBLE, N_EXPECTED_EXCLUDED,
                        sha256_file, atomic_write_json, windows_process_creation_time)


def src(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


config = {
    "candidate_version": CANDIDATE_VERSION,
    "track": TRACK,
    "batch": BATCH,
    "parent_of_this": "v6.11.0-sfeatgap.rc1 (batch10 full 1069-feature fit; loaded/replayed, never retrained)",
    "baseline_of_this": "v6.10.0-starget.rc1 (batch09 45-feature raw-S Ridge/HGB; loaded/replayed, never retrained)",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "freeze_note": ("config.json written BEFORE any batch11 held-out (val/test) metric "
                    "was computed; immutable freeze-point for the four predeclared "
                    "scientific fits (A=Ridge, A=HGB, B=Ridge, B=HGB on raw S)."),
    "sources_sha256": {
        "gene_expression_table.tsv": src(EXPR),
        "gene_expression_label_table.tsv": src(CANON),
        "batch08_label_audit_all_genes.tsv": src(AUDIT_TSV),
        "batch03_trackA_gene_promoters.npz": src(TRACKA_NPZ),
        "batch05_split_arrays.npz": src(SPLIT_NPZ),
        "batch06_features45.npz": src(FEATURES_NPZ),
        "promoter_sym400_uint8.memmap": src(MEM400),
        "promoter_up2000_uint8.memmap": src(MEM2000),
        "batch10_features1024.npz": src(B10_FEATURES1024),
    },
    "expected_source_sha256": {
        "batch05_split_arrays.npz": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
        "batch06_features45.npz": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
        "batch03_trackA_gene_promoters.npz": "cd53ccb608c2969f279ac46ed49f94c9b56efd672393c937ec730139e61e4072",
        "batch10_features1024.npz": "2dc7dc0eda7c60b3fa740261e39fb37832dc0e0faf81a7fa2f9cd37a11bc14f2",
    },
    "population": {
        "canonical_genes": N_CANONICAL,
        "canonical_S_defined": M_DEFINED,
        "feature_universe_genes": N_FEATURE_UNIVERSE,
        "expected_eligible_S_defined_and_feature_covered": N_EXPECTED_ELIGIBLE,
        "expected_excluded_canonical_S_defined": N_EXPECTED_EXCLUDED,
        "expected_components_total": 18400,
        "note": ("Eligible = canonical S-defined AND present in the batch03 57827-gene "
                 "promoter-feature universe (has frozen composition/count features and a "
                 "batch05 component split). The 178 excluded genes are S-defined in the "
                 "canonical expression table but absent from the promoter-feature universe."),
    },
    "S_label_definition": {
        "DETECTABLE_TPM": DETECTABLE_TPM,
        "B_MIN": B_MIN,
        "MIN_DET_TISSUES": MIN_DET_TISSUES,
        "formula": "S_raw = -MAD(log2(1+collapsed_mean_tpm)) over detectable (TPM>=1) tissues",
        "eligibility": "B_raw = detectable/54 >= 0.30 AND detectable_tissue_count >= 3",
        "direction": "larger / less-negative S = less across-tissue dispersion",
        "not": "S is NOT post-transfection temporal stability nor synthetic promoter activity",
        "reference_target": "full-precision raw S (reconstructed in memory from gene_expression_table.tsv)",
        "frozen_target_only": "continuous raw S only; do NOT re-run raw/rank comparison",
    },
    "split": {
        "source": "data/breakthrough_20261002/batch05/split_arrays.npz",
        "name": "exact/RC + detected-near-identity blocked component split (NOT homology-free)",
        "seed": SEED,
        "unchanged": True,
        "note": ("component-level split reused exactly as frozen; retrospective not prospective; "
                 "exact/RC isolation does not establish complete near-homology independence."),
    },
    "sequence_encoding_conventions": {
        "uint8_code": {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4},
        "sym400_window": "promoter_sym400_uint8.memmap row, length 400, bins [0,100),[100,200),[200,300),[300,400)",
        "up2000_window": "promoter_up2000_uint8.memmap row, length 2000, bins [0,500),[500,1000),[1000,1500),[1500,2000)",
        "r400_r2000": "row indices into the corresponding memmaps (verified against promoter_*_index.tsv)",
        "strand_orientation": "preserved exactly; NO reverse-complement canonicalization",
        "gene_identity": "gene_id[starts[:-1]] (first row of each gene group), NOT gene_id[g]",
        "promoter_aggregation": "normalize within each promoter first, then arithmetic mean across ALL promoters of the gene",
        "no_label_driven_promoter_selection": True,
    },
    "features": {
        "baseline_source": "data/breakthrough_20261002/batch06/features45.npz",
        "baseline_n_features": 45,
        "baseline_feature_names": FEATURE_NAMES_45,
        "new_feature_source": "data/breakthrough_20261002/batch10/attempt_001_extract/features1024.npz",
        "new_feature_n": 1024,
        "feature_names_1024": FEATURE_NAMES_1024,
        "feature_names_all": FEATURE_NAMES_ALL,
        "n_features_full": 1069,
        "arm_A": {
            "name": "old45 + positional3mer512",
            "definition": "45 baseline + all 512 local-position trinucleotide features (sym400 4x64 + up2000 4x64)",
            "indices_1024": list(range(0, 512)),
            "n_features": 557,
        },
        "arm_B": {
            "name": "old45 + global4mer512",
            "definition": "45 baseline + all 512 full-window tetranucleotide features (sym400 256 + up2000 256)",
            "indices_1024": list(range(512, 1024)),
            "n_features": 557,
        },
        "A_B_differ_note": ("A and B differ BOTH in k-mer length AND in positional aggregation; "
                            "differences cannot be uniquely attributed to position alone."),
        "verification_by_name": "dictionary/order verified by feature names, not blind slicing",
        "no_label_features": True,
        "no_new_features_or_k5_or_motif": True,
    },
    "research_fits": {
        "count": 4,
        "arm_A_ridge": {"model": "Ridge", "config": RIDGE, "features": "old45+position3mer512 (557)", "target": "continuous raw S (standardized, train-only)"},
        "arm_A_hgb": {"model": "HistGradientBoostingRegressor", "config": HGB, "features": "old45+position3mer512 (557)", "target": "continuous raw S (standardized, train-only)"},
        "arm_B_ridge": {"model": "Ridge", "config": RIDGE, "features": "old45+global4mer512 (557)", "target": "continuous raw S (standardized, train-only)"},
        "arm_B_hgb": {"model": "HistGradientBoostingRegressor", "config": HGB, "features": "old45+global4mer512 (557)", "target": "continuous raw S (standardized, train-only)"},
        "no_tuning_feature_selection_extra_seeds_ensemble": True,
        "validation_metrics_descriptive_only": True,
    },
    "full_reference": {
        "source": "data/breakthrough_20261002/batch10/predictions.tsv",
        "ridge_col": "ridge_aug_pred",
        "hgb_col": "hgb_aug_pred",
        "role": "reference for the four ablation contrasts; LOADED/REPLAYED, never retrained",
        "not_an_upper_bound": "full 1069 model is a reference, NOT a mathematical upper bound",
    },
    "baseline_reuse": {
        "batch09_ridge_raw_test_spearman": 0.37008663146628534,
        "batch09_hgb_raw_test_spearman": 0.40888782012905495,
        "source": "data/breakthrough_20261002/batch09/predictions.tsv (ridge_raw_pred / hgb_raw_pred)",
        "replay": "verify loaded/reproducible without refitting (exact old-prediction matching)",
        "role": "secondary descriptive comparison for A/B arms, not an extra confirmatory discovery",
    },
    "transform_rules": {
        "X_scaler": "sklearn StandardScaler (with_mean/with_std) fit on eligible TRAIN rows only, per representation (A and B each own scaler); reused for val/test",
        "target_standardization": "z = (y - train_mean) / train_std (population std, ddof=0), fit on eligible TRAIN only",
        "target_inverse": "y_hat = z_hat * train_std + train_mean (back to raw S units)",
    },
    "evaluation": {
        "seed": SEED,
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_unit": "test components (genes within a component stay together)",
        "bootstrap_paired": "full and ablated arm evaluated on the SAME resample; delta = full Spearman - arm Spearman",
        "interval": "percentile interval [0.625, 99.375] percent => 98.75 percent two-sided per comparison",
        "interval_label": "Bonferroni-style 95 percent family convention across FOUR comparisons (each uses a 98.75 percent two-sided percentile interval)",
        "acceptance_rule": "scoped conditional increment requires delta >= %.2f AND adjusted lower bound > 0" % DELTA_MIN,
        "primary_comparisons": [
            "Ridge: (full - A) test Spearman => incremental contribution of B given A",
            "Ridge: (full - B) test Spearman => incremental contribution of A given B",
            "HGB: (full - A) test Spearman => incremental contribution of B given A",
            "HGB: (full - B) test Spearman => incremental contribution of A given B",
        ],
        "all_four_shown": True,
        "no_posthoc_selection": True,
        "also_report": "raw-unit MAE/MSE/R2 and constant baseline; R2 is calculated, never inferred from rho",
        "secondary_descriptive_only": "A/B versus old45 baseline are descriptive, NOT extra confirmatory discoveries",
        "not_equivalence": "failure to detect incremental value is NOT proof of equivalence or redundancy",
    },
    "resources": {
        "cpu_threads_max": 4,
        "gpu": "zero",
        "fits": "four scientific fits (A_Ridge, A_HGB, B_Ridge, B_HGB)",
        "numerical_budget_minutes": 45,
        "ram_target_gb": 4,
        "no_search": "no hyperparameter / architecture / threshold / target-mapping search",
        "no_validation_fit": "no val/test fitting of scaler, hyperparameter, stopping rule or target mapping",
    },
    "scope_limits": [
        "human conditional gene-level retrospective raw S prediction only",
        "not mammalian universal promoter strength / synthetic promoter activity / HEK293T construct validation / temporal stability",
        "no prospective or causal claims",
        "reused fixed test split has adaptive research exposure; repeated test use",
        "incomplete near-homology isolation remains",
        "this is predictive ablation, NOT causal mechanism identification",
        "A vs B differs in BOTH k and positional aggregation; no unique position attribution",
        "standalone gains versus baseline do NOT prove independent/additive contributions",
        "full model is a reference, not a mathematical upper bound",
        "raw-vs-rank is nonlinear, not generally equivalent rescaling; no rerun here (outside this question)",
    ],
}


if __name__ == "__main__":
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MODEL_OUT / "config.json", config)
    print("wrote", MODEL_OUT / "config.json")

