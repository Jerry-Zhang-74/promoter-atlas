#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 v6.14.0-scv5.rc1 frozen config.json (written BEFORE any fit/heldout metric).

Immutable freeze-point for the component five-fold cross-fitting retrospective
robustness experiment. Records the frozen population, the label-independent fold
allocation algorithm, the single HGB model config, the paired per-fold scaling
rules, the bootstrap convention and the predeclared acceptance criterion.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b13_common import (  # noqa: E402
    ROOT, EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ, FEATURES45_NPZ,
    FEATURES1024_NPZ, MODEL_OUT, CANDIDATE_VERSION, TRACK, BATCH, HGB, FIT_SEED,
    BOOT_SEED, N_FOLDS, N_BOOTSTRAP, BOOT_LO_PCT, BOOT_HI_PCT, DELTA_MIN,
    MIN_POSITIVE_FOLDS, FOLD_SALT, DETECTABLE_TPM, B_MIN, MIN_DET_TISSUES,
    N_CANONICAL, M_DEFINED, N_FEATURE_UNIVERSE, N_EXPECTED_ELIGIBLE,
    N_EXPECTED_EXCLUDED, N_EXPECTED_COMPONENTS, sha256_file, atomic_write_json)


def src(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def main():
    config = {
        "candidate_version": CANDIDATE_VERSION,
        "track": TRACK,
        "batch": BATCH,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "freeze_note": (
            "config.json written BEFORE any batch13 fit or held-out metric was "
            "computed; immutable freeze-point for the component five-fold "
            "cross-fitting retrospective robustness experiment."
        ),
        "sources_sha256": {
            "gene_expression_table.tsv": src(EXPR),
            "gene_expression_label_table.tsv": src(CANON),
            "batch08_label_audit_all_genes.tsv": src(AUDIT_TSV),
            "batch03_trackA_gene_promoters.npz": src(TRACKA_NPZ),
            "batch05_split_arrays.npz": src(SPLIT_NPZ),
            "batch06_features45.npz": src(FEATURES45_NPZ),
            "batch10_features1024.npz": src(FEATURES1024_NPZ),
        },
        "expected_source_sha256": {
            "split_arrays.npz": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
            "features45.npz": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
            "features1024.npz": "2dc7dc0eda7c60b3fa740261e39fb37832dc0e0faf81a7fa2f9cd37a11bc14f2",
            "trackA_gene_promoters.npz": "cd53ccb608c2969f279ac46ed49f94c9b56efd672393c937ec730139e61e4072",
        },
        "population": {
            "canonical_genes": N_CANONICAL,
            "canonical_S_defined": M_DEFINED,
            "feature_universe_genes": N_FEATURE_UNIVERSE,
            "expected_eligible_S_defined_and_feature_covered": N_EXPECTED_ELIGIBLE,
            "expected_excluded_canonical_S_defined": N_EXPECTED_EXCLUDED,
            "expected_eligible_components": N_EXPECTED_COMPONENTS,
            "note": (
                "Eligible = canonical S-defined AND present in the batch03 57827-gene "
                "promoter-feature universe (has frozen composition/count features and a "
                "batch05 component split). 18400 distinct component IDs among eligible."
            ),
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
            "frozen_target_only": "continuous raw S only",
        },
        "fold_allocation": {
            "scheme": "component five-fold cross-fitting",
            "label_independent": True,
            "labels_not_used": "raw S values never used for fold assignment",
            "algorithm": (
                "1) collect unique component IDs of eligible genes; "
                "2) sort them by canonical string representation; "
                "3) compute hash = SHA256(FOLD_SALT + ID) with FOLD_SALT = %s; "
                "4) sort components by decreasing eligible gene size, then by hash, "
                "then by ID; "
                "5) greedy-assign each component (in that order) to the fold with the "
                "smallest current eligible gene count (ties -> lowest fold index 0..4)."
            ) % FOLD_SALT,
            "salt": FOLD_SALT,
            "n_folds": N_FOLDS,
            "every_gene_exactly_once_in_heldout": True,
            "no_component_crossing": True,
            "original_batch05_split_roles_retained_as_metadata_only": True,
            "no_original_test_heldout_claim": True,
        },
        "model": {
            "model": "HistGradientBoostingRegressor",
            "config": HGB,
            "baseline_arm": "45 features (batch06 composition/count)",
            "full_arm": "1069 features (45 baseline + 1024 batch10 positional sequence)",
            "n_scientific_fits": 10,
            "fits_per_fold": 2,
            "no_feature_selection_hyperparam_search_seed_search_ensemble_gpu": True,
            "target": "continuous raw S only",
        },
        "transform_rules": {
            "X_scaler": "sklearn StandardScaler (with_mean/with_std) fit on the other four folds only; independent scaler per representation (base vs full)",
            "target_standardization": "z = (y - train_mean) / train_std (population std ddof=0) fit on the other four folds only; shared within a paired fold",
            "target_inverse": "y_hat = z_hat * train_std + train_mean (back to raw S units)",
            "no_original_batch10_scaler_or_model_reuse": "original batch10 scalers/models saw different rows; NEVER reused for cross-fit training",
            "feature_extraction_reused": "frozen batch06/batch10 feature matrices only; no refit of feature extraction",
        },
        "evaluation": {
            "seed": BOOT_SEED,
            "bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_unit": "components resampled WITHIN fold, preserving per-fold component counts",
            "bootstrap_paired": "full and base evaluated on the SAME resample; delta = full Spearman - base Spearman over pooled resampled OOF rows",
            "interval": "percentile interval [2.5, 97.5] percent => 95 percent two-sided",
            "primary_contrast": "pooled OOF full-minus-base Spearman",
            "acceptance_criterion": (
                "pooled delta >= %.2f AND 95 percent lower bound > 0 AND "
                "positive fold Spearman delta in >= %d of %d folds"
            ) % (DELTA_MIN, MIN_POSITIVE_FOLDS, N_FOLDS),
            "fixed_prediction_bootstrap_scope": (
                "captures held-out sampling variability conditional on these ten fits; "
                "NOT training variation and NOT full research-selection/adaptivity uncertainty"
            ),
            "show_all_folds": True,
            "raw_r2_mae_mse_deltas_descriptive_secondary": True,
        },
        "scope_limits": [
            "retrospective robustness on reused labels; NOT independent external or prospective validation",
            "does NOT repair all near-homology / adaptive-test concerns",
            "exact/RC component grouping remains incomplete for near homology",
            "positive threshold supports representation robustness within these conditions only",
            "not reliable HEK293T / fragment activity / temporal stability / causal mechanism evidence",
        ],
        "resources": {
            "cpu_threads_max": 4,
            "gpu": "zero",
            "fits": "exactly ten scientific fits (2 per fold x 5 folds)",
            "numerical_fit_eval_budget_minutes": 60,
            "no_search": "no hyperparameter / architecture / threshold / target-mapping search",
        },
    }
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MODEL_OUT / "config.json", config)
    print("wrote", MODEL_OUT / "config.json")


if __name__ == "__main__":
    main()
