#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the frozen batch09 config.json BEFORE any held-out metric is computed.

This is a freeze-point artifact: source/split/feature/label SHAs, all constants,
transform formulas, model configs and the evaluation convention. It must exist
and be immutable before the four research fits are run or any val/test number
is inspected.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b9_common import (ROOT, EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ,
                       FEATURES_NPZ, MODEL_OUT, CANDIDATE_VERSION, TRACK, BATCH,
                       SEED, N_BOOTSTRAP, BOOT_LO_PCT, BOOT_HI_PCT, DELTA_MIN,
                       RIDGE, HGB, AFF_A, AFF_B, FEATURE_NAMES,
                       DETECTABLE_TPM, B_MIN, MIN_DET_TISSUES,
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
    "freeze_note": ("config.json written BEFORE any batch09 held-out (val/test) "
                    "metric was computed; immutable freeze-point for the four "
                    "predeclared research fits."),
    "sources_sha256": {
        "gene_expression_table.tsv": src(EXPR),
        "gene_expression_label_table.tsv": src(CANON),
        "batch08_label_audit_all_genes.tsv": src(AUDIT_TSV),
        "batch03_trackA_gene_promoters.npz": src(TRACKA_NPZ),
        "batch05_split_arrays.npz": src(SPLIT_NPZ),
        "batch06_features45.npz": src(FEATURES_NPZ),
    },
    "expected_source_sha256": {
        "batch05_split_arrays.npz": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
        "batch06_features45.npz": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
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
        "not": "S is NOT post-transfection temporal stability",
        "reference_target": "full-precision raw S (reconstructed in memory from gene_expression_table.tsv)",
        "tie_note": ("full-precision raw S has 0 true ties across the 18947 defined genes; "
                     "the 6-decimal audit export and 10g export do not change ordering "
                     "(verified). See preflight for measured tie/ordering checks."),
    },
    "split": {
        "source": "data/breakthrough_20261002/batch05/split_arrays.npz",
        "name": "exact/RC + detected-near-identity blocked component split (NOT homology-free)",
        "seed": 20261011,
        "unchanged": True,
        "note": ("component-level split reused exactly as frozen; retrospective not prospective; "
                 "exact/RC isolation does not establish complete near-homology independence."),
    },
    "features": {
        "source": "data/breakthrough_20261002/batch06/features45.npz",
        "n_features": 45,
        "composition": "comp44 (sym400+up2000 mono/dinuc/GC/length = 44) + log1p_count (1)",
        "feature_names": FEATURE_NAMES,
        "no_label_features": True,
        "note": "composition/count only; no E/S/B labels used as features.",
    },
    "research_arms": {
        "arm_ridge_raw": {"model": "Ridge", "config": RIDGE, "target": "continuous raw S (standardized)"},
        "arm_ridge_rank": {"model": "Ridge", "config": RIDGE, "target": "train-reference empirical rank F (standardized)"},
        "arm_hgb_raw": {"model": "HistGradientBoostingRegressor", "config": HGB, "target": "continuous raw S (standardized)"},
        "arm_hgb_rank": {"model": "HistGradientBoostingRegressor", "config": HGB, "target": "train-reference empirical rank F (standardized)"},
    },
    "transform_rules": {
        "X_scaler": "sklearn StandardScaler (with_mean/with_std) fit on eligible TRAIN rows only; reused for val/test",
        "target_standardization": "z = (y - train_mean) / train_std (population std, ddof=0), fit on eligible TRAIN only",
        "target_inverse": "y_hat = z_hat * train_std + train_mean (back to target units)",
        "rank_formula": "F(x) = (count(train < x) + 0.5 * count(train == x)) / n_train; applied as a frozen function to train/val/test; values outside train support map to 0 (below) or 1 (above)",
        "rank_reference_note": "train-reference empirical rank; NOT the global canonical rank",
        "common_ranking_truth": "full-precision raw S (regardless of arm)",
    },
    "evaluation": {
        "seed": SEED,
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_unit": "test components (genes within a component stay together)",
        "bootstrap_paired": "both arms evaluated on the SAME resample; delta = rank Spearman - raw Spearman",
        "interval": "percentile interval [1.25, 98.75] percent => 97.5 percent two-sided per comparison",
        "interval_label": ("conservative two-comparison adjustment convention (two primary comparisons); "
                           "each comparison uses a 97.5 percent two-sided percentile interval"),
        "acceptance_rule": "scoped target-representation improvement requires delta >= 0.03 AND adjusted lower bound > 0",
        "primary_comparisons": [
            "rank-target minus raw-target TEST Spearman within Ridge",
            "rank-target minus raw-target TEST Spearman within HGB",
        ],
        "descriptive_also": "fixed Ridge vs HGB within the same conditional test rows",
        "no_cross_unit_mae": "MAE magnitudes are NEVER compared between raw and rank units",
        "no_out_of_scope_comparison": "do NOT compare these S results to batch06 E performance or a v6.3 result",
    },
    "affine_diagnostic": {
        "kind": "algebra test (not a fifth research arm, not a score gain)",
        "transform": "y2 = AFF_A + AFF_B * y",
        "AFF_A": AFF_A,
        "AFF_B": AFF_B,
        "model": "Ridge (identical alpha/intercept as research Ridge)",
        "expectation": "predictions affine-related within numeric tolerance",
        "metrics": "paired affine R2/Spearman invariance; unit-dependent MAE (x b) / MSE (x b^2)",
    },
    "resources": {
        "cpu_threads_max": 4,
        "gpu": "zero",
        "fits": "four research fits + one affine diagnostic (algebra test)",
        "no_search": "no hyperparameter / architecture / threshold / target-mapping search",
        "no_validation_fit": "no val/test fitting of scaler, CDF, hyperparameter, stopping rule or target mapping",
    },
    "scope_limits": [
        "human conditional gene-level S prediction only",
        "not mammalian universal promoter strength / synthetic promoter activity / HEK293T construct validation / temporal stability",
        "no prospective or causal claims",
        "n=1 input equality never implies separately trained models have identical outputs",
        "shared-trunk missing-label masking does not exclude indirect training influence from other labels",
    ],
}


if __name__ == "__main__":
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MODEL_OUT / "config.json", config)
    print("wrote", MODEL_OUT / "config.json")
