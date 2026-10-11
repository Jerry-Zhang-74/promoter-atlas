#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the frozen batch10 config.json BEFORE any held-out metric is computed.

Immutable freeze-point for the two predeclared scientific fits (augmented Ridge
and augmented HGB on continuous raw S). Records source/split/feature/label SHAs,
encoding/window conventions, all constants, transform formulas, model configs
and the evaluation convention. Written once before extraction results are
inspected for val/test numbers.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b10_common import (ROOT, EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ,
                        FEATURES_NPZ, MEM400, MEM2000, MODEL_OUT, CANDIDATE_VERSION,
                        TRACK, BATCH, SEED, N_BOOTSTRAP, BOOT_LO_PCT, BOOT_HI_PCT,
                        DELTA_MIN, RIDGE, HGB, FEATURE_NAMES_45, FEATURE_NAMES_1024,
                        FEATURE_NAMES_ALL, DETECTABLE_TPM, B_MIN, MIN_DET_TISSUES,
                        N_CANONICAL, M_DEFINED, N_FEATURE_UNIVERSE,
                        N_EXPECTED_ELIGIBLE, N_EXPECTED_EXCLUDED,
                        sha256_file, atomic_write_json)


def src(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


config = {
    "candidate_version": CANDIDATE_VERSION,
    "track": TRACK,
    "batch": BATCH,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "freeze_note": ("config.json written BEFORE any batch10 held-out (val/test) metric "
                    "was computed; immutable freeze-point for the two predeclared "
                    "scientific fits (augmented Ridge and augmented HGB on raw S)."),
    "sources_sha256": {
        "gene_expression_table.tsv": src(EXPR),
        "gene_expression_label_table.tsv": src(CANON),
        "batch08_label_audit_all_genes.tsv": src(AUDIT_TSV),
        "batch03_trackA_gene_promoters.npz": src(TRACKA_NPZ),
        "batch05_split_arrays.npz": src(SPLIT_NPZ),
        "batch06_features45.npz": src(FEATURES_NPZ),
        "promoter_sym400_uint8.memmap": src(MEM400),
        "promoter_up2000_uint8.memmap": src(MEM2000),
    },
    "expected_source_sha256": {
        "batch05_split_arrays.npz": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
        "batch06_features45.npz": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
        "batch03_trackA_gene_promoters.npz": "cd53ccb608c2969f279ac46ed49f94c9b56efd672393c937ec730139e61e4072",
    },
    "population": {
        "canonical_genes": N_CANONICAL,
        "canonical_S_defined": M_DEFINED,
        "feature_universe_genes": N_FEATURE_UNIVERSE,
        "expected_eligible_S_defined_and_feature_covered": N_EXPECTED_ELIGIBLE,
        "expected_excluded_canonical_S_defined": N_EXPECTED_EXCLUDED,
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
        "new_feature_block": {
            "description": ("exactly 1024 new features concatenated to the unchanged 45-feature "
                            "baseline (total 1069)"),
            "tri_bin_features": "2 windows x 4 bins x 64 oriented contiguous trinucleotide frequencies",
            "tetra_features": "2 windows x 256 oriented contiguous tetranucleotide frequencies",
            "kmer_order": "lexicographic A,C,G,T (base-4 index A=0,C=1,G=2,T=3)",
            "valid_start_denominator": "number of wholly-valid ACGT k-mer starts inside the block/window",
            "ambiguous_handling": "ambiguous bases break k-mers; no valid starts => all-zero block (frequencies 0), never NaN",
            "no_bin_boundary_double_count": True,
            "no_k5_or_motif_library_or_extra_features": True,
        },
        "feature_names_1024": FEATURE_NAMES_1024,
        "feature_names_all": FEATURE_NAMES_ALL,
        "n_features_total": 1069,
        "no_label_features": True,
    },
    "research_fits": {
        "count": 2,
        "arm_ridge_aug": {
            "model": "Ridge",
            "config": RIDGE,
            "target": "continuous raw S (standardized, train-only)",
            "features": "1069 (45 baseline + 1024 new)",
        },
        "arm_hgb_aug": {
            "model": "HistGradientBoostingRegressor",
            "config": HGB,
            "target": "continuous raw S (standardized, train-only)",
            "features": "1069 (45 baseline + 1024 new)",
        },
        "no_tuning_feature_selection_extra_seeds_ensemble": True,
        "validation_metrics_descriptive_only": True,
    },
    "transform_rules": {
        "X_scaler": "sklearn StandardScaler (with_mean/with_std) fit on eligible TRAIN rows only; reused for val/test",
        "target_standardization": "z = (y - train_mean) / train_std (population std, ddof=0), fit on eligible TRAIN only",
        "target_inverse": "y_hat = z_hat * train_std + train_mean (back to raw S units)",
    },
    "baseline_reuse": {
        "batch09_ridge_raw_test_spearman": 0.37008663146628534,
        "batch09_hgb_raw_test_spearman": 0.40888782012905495,
        "source": "data/breakthrough_20261002/batch09/predictions.tsv",
        "replay": "verify saved batch09 models/scaler/transforms reproduce ridge_raw/hgb_raw predictions on a deterministic small subset (no refit)",
    },
    "evaluation": {
        "seed": SEED,
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_unit": "test components (genes within a component stay together)",
        "bootstrap_paired": "augmented and baseline evaluated on the SAME resample; delta = aug Spearman - baseline Spearman",
        "interval": "percentile interval [1.25, 98.75] percent => 97.5 percent two-sided per comparison",
        "interval_label": "conservative two-comparison adjustment convention (two primary comparisons)",
        "acceptance_rule": "scoped feature improvement requires delta >= 0.03 AND adjusted lower bound > 0",
        "primary_comparisons": [
            "augmented-minus-baseline TEST Spearman within Ridge",
            "augmented-minus-baseline TEST Spearman within HGB",
        ],
        "also_report": "raw-unit MAE/MSE/R2 deltas and constant baselines; R2 is calculated, never inferred from rho",
        "no_cross_unit_mae": "MAE magnitudes never compared between raw and any other unit",
        "no_out_of_scope_comparison": "do NOT compare these S results to batch06 E performance or a v6.3 result",
    },
    "resources": {
        "cpu_threads_max": 4,
        "gpu": "zero",
        "fits": "two scientific fits (augmented Ridge + augmented HGB)",
        "no_search": "no hyperparameter / architecture / threshold / target-mapping search",
        "no_validation_fit": "no val/test fitting of scaler, hyperparameter, stopping rule or target mapping",
    },
    "scope_limits": [
        "human conditional gene-level S prediction only",
        "not mammalian universal promoter strength / synthetic promoter activity / HEK293T construct validation / temporal stability",
        "no prospective or causal claims",
        "reused fixed test split has adaptive research exposure; repeated test use",
        "incomplete near-homology isolation remains",
        "positive result supports only this specific feature addition; negative result rejects only this tested representation/configuration",
    ],
}


if __name__ == "__main__":
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MODEL_OUT / "config.json", config)
    print("wrote", MODEL_OUT / "config.json")
