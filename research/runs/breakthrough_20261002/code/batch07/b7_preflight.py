#!/usr/bin/env python3
"""Batch07 preflight: verify frozen sources, unique IDs, feature contract, joins,
and bounded CPU replay of the frozen HGB / ridge predictors against saved outputs.
"""
import sys
import json
import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b7_common import (  # noqa: E402
    REPORT, MODEL_DIR, DATA05, DATA06, MODEL_DIR06, DATA03,
    atomic_write_json, sha256_file, CANDIDATE,
    SPLIT_ARRAYS_SHA, FEATURES45_SHA, HGB_SHA, FINALRIDGE_SHA, OOF_RESIDUALS_SHA,
    PRIMARY_QUANTILE, DIAGNOSTIC_QUANTILES, QUANTILE_METHOD,
    MIN_TEST_COVERAGE, MIN_REL_MAE_REDUCTION, BOOT_N, BOOT_SEED,
    RANDOM_MASK_N, RANDOM_MASK_SEED, HIGH_ERROR_THRESHOLD,
)
from b7_data import load_labels_split, load_features, load_saved_predictions, build_split_rows  # noqa: E402

LABEL_TABLE = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"


def main():
    checks = {}
    gene_gid, gene_score_e, gene_split, comp_dense, n_prom, n_genes = load_labels_split()
    X45, comp44, log1p_count, gc_gene = load_features()

    checks["split_arrays_sha"] = sha256_file(DATA05 / "split_arrays.npz")
    checks["split_arrays_sha_ok"] = checks["split_arrays_sha"] == SPLIT_ARRAYS_SHA
    checks["features45_sha"] = sha256_file(DATA06 / "features45.npz")
    checks["features45_sha_ok"] = checks["features45_sha"] == FEATURES45_SHA
    checks["hgb_sha"] = sha256_file(MODEL_DIR06 / "hgb_baseline.joblib")
    checks["hgb_sha_ok"] = checks["hgb_sha"] == HGB_SHA
    checks["finalridge_sha"] = sha256_file(MODEL_DIR06 / "finalridge.npz")
    checks["finalridge_sha_ok"] = checks["finalridge_sha"] == FINALRIDGE_SHA
    checks["oof_residuals_sha"] = sha256_file(DATA06 / "oof_residuals.npz")
    checks["oof_residuals_sha_ok"] = checks["oof_residuals_sha"] == OOF_RESIDUALS_SHA

    checks["n_genes"] = int(n_genes)
    checks["gene_gid_unique"] = bool(len(np.unique(gene_gid)) == n_genes)
    checks["gene_gid_len"] = bool(len(gene_gid) == n_genes)
    checks["features_rows"] = bool(X45.shape[0] == n_genes)
    checks["features_cols"] = int(X45.shape[1])
    checks["features_cols_ok"] = bool(X45.shape[1] == 45)
    checks["features_no_nan_inf"] = bool(np.isfinite(X45).all())
    checks["split_counts"] = {s: int((gene_split == s).sum()) for s in ["train", "val", "test"]}
    checks["split_total_ok"] = bool(int(sum(checks["split_counts"].values())) == n_genes)
    checks["n_prom_all_positive"] = bool(np.all(n_prom >= 1))
    checks["n_prom_len_ok"] = bool(len(n_prom) == n_genes)
    checks["comp_dense_len_ok"] = bool(len(comp_dense) == n_genes)

    lab = np.genfromtxt(LABEL_TABLE, delimiter="\t", names=True, dtype=None, encoding="utf-8")
    lab_map = {str(r["gene_id"]): float(r["Score_E"]) for r in lab}
    matched = 0
    max_abs_diff = 0.0
    for gid, val in zip(gene_gid, gene_score_e):
        if gid in lab_map:
            matched += 1
            max_abs_diff = max(max_abs_diff, abs(lab_map[gid] - float(val)))
    checks["label_table_matched"] = int(matched)
    checks["label_table_max_abs_diff"] = float(max_abs_diff)
    checks["label_table_diff_tiny"] = bool(max_abs_diff < 1e-4)

    b6, b5r_map, b5b2_map = load_saved_predictions()
    join = {"n_missing": 0, "n_matched": 0, "n_rows_val": 0, "n_rows_test": 0}
    for split_name in ["val", "test"]:
        gid, y, comp, nsub, idx, _ = build_split_rows(split_name)
        join["n_rows_" + split_name] = int(len(gid))
        for g, yy in zip(gid, y):
            if g in b6:
                join["n_matched"] += 1
            else:
                join["n_missing"] += 1
    checks["join"] = join
    checks["join_no_missing"] = bool(join["n_missing"] == 0)
    checks["join_matched_total"] = int(join["n_matched"])

    import joblib
    hgb = joblib.load(MODEL_DIR06 / "hgb_baseline.joblib")
    hgb_replay = hgb.predict(X45)

    fr = np.load(MODEL_DIR06 / "finalridge.npz")
    coef = fr["coef"]; intercept = fr["intercept"]; mu = fr["mu"]; sd = fr["sd"]
    fr_replay = ((X45 - mu) / sd) @ coef + intercept

    replay = {"hgb_max_abs_diff": 0.0, "ridge_max_abs_diff": 0.0, "hgb_n_checked": 0, "ridge_n_checked": 0}
    for split_name in ["val", "test"]:
        gid, y, comp, nsub, idx, _ = build_split_rows(split_name)
        for i, g in enumerate(gid):
            gi = int(idx[i])
            hd = abs(float(b6[g]["hgb"]) - float(hgb_replay[gi]))
            rd = abs(float(b6[g]["finalridge"]) - float(fr_replay[gi]))
            replay["hgb_max_abs_diff"] = max(replay["hgb_max_abs_diff"], hd)
            replay["ridge_max_abs_diff"] = max(replay["ridge_max_abs_diff"], rd)
            replay["hgb_n_checked"] += 1
            replay["ridge_n_checked"] += 1
    checks["replay"] = replay
    checks["replay_hgb_exact"] = bool(replay["hgb_max_abs_diff"] < 1e-9)
    checks["replay_ridge_exact"] = bool(replay["ridge_max_abs_diff"] < 1e-9)

    source_shas = {
        "split_arrays_npz": checks["split_arrays_sha"],
        "features45_npz": checks["features45_sha"],
        "hgb_baseline_joblib": checks["hgb_sha"],
        "finalridge_npz": checks["finalridge_sha"],
        "oof_residuals_npz": checks["oof_residuals_sha"],
        "trackA_gene_promoters_npz": sha256_file(DATA03 / "trackA_gene_promoters.npz"),
        "label_table_tsv": sha256_file(LABEL_TABLE),
    }

    config = {
        "candidate_version": CANDIDATE,
        "question": "Can ONE pre-registered train-only feature-coverage rule reduce selective-prediction error at high coverage for the frozen HGB model? (applicability diagnostic, not certified confidence)",
        "scope": "gene-level Score_E ranking only; NOT promoter/construct/HEK293T activity",
        "retrospective_note": "existing split/test repeatedly reused => retrospective/exploratory only, never prospective; a train-only threshold does not erase prior outcome-informed project decisions",
        "predictors": {
            "primary": "frozen batch06 HGB (sklearn HistGradientBoostingRegressor)",
            "reference": ["frozen finalridge", "batch05 B2 saved predictions"],
            "optional_descriptive": "batch06 combined_fixedmean (saved, no CNN recompute)",
        },
        "coverage_rule": {
            "geometry": "StandardScaler fit on outer-train features only, then sklearn LedoitWolf covariance on standardized train features",
            "coverage_score": "squared Mahalanobis distance (x-mu)^T precision (x-mu); lower = more central in this particular feature geometry, NOT calibrated probability",
            "primary_threshold": "90th percentile of TRAIN distances (deterministic linear quantile)",
            "diagnostic_points": [str(int(q * 100)) + "%" for q in DIAGNOSTIC_QUANTILES],
            "quantile_method": QUANTILE_METHOD,
            "no_labels_errors_heldout_used": True,
            "feature_centrality_not_density": "multimodality and data-internal annotation bias remain",
        },
        "acceptance_primary": {
            "min_test_coverage": MIN_TEST_COVERAGE,
            "min_rel_mae_reduction": MIN_REL_MAE_REDUCTION,
            "component_bootstrap_lb_gt_0": "whole-component bootstrap 95% LB for allMAE - retainedMAE > 0",
            "bootstrap_n": BOOT_N,
            "bootstrap_seed": BOOT_SEED,
            "random_mask_n": RANDOM_MASK_N,
            "random_mask_seed": RANDOM_MASK_SEED,
            "high_error_threshold": HIGH_ERROR_THRESHOLD,
        },
    }

    all_pass = all(v is not False for v in checks.values() if isinstance(v, bool))
    out = {
        "checks": checks,
        "source_shas": source_shas,
        "config": config,
        "all_pass": all_pass,
        "written_at": datetime.datetime.now().astimezone().isoformat(),
    }
    atomic_write_json(REPORT / "preflight_b7.json", out)
    atomic_write_json(REPORT / "frozen_config_b7.json", config)

    print(json.dumps(checks, indent=2, ensure_ascii=False, default=float))
    print("all_pass =", all_pass)
    if not all_pass:
        print("PREFLIGHT FAILED")
        sys.exit(1)
    print("PREFLIGHT PASS")


if __name__ == "__main__":
    main()
