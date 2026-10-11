#!/usr/bin/env python3
"""Batch07 tests: train-only threshold fit, no-label geometry, equal row joins,
inference structure, scope refusals, and random-mask baseline reproducibility.
"""
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b7_common import (  # noqa: E402
    MODEL_DIR, REPORT, sha256_file, mahalanobis_sq, quantile_linear, mae,
    PRIMARY_QUANTILE, DIAGNOSTIC_QUANTILES, RANDOM_MASK_N, RANDOM_MASK_SEED,
    CANDIDATE,
)
from b7_data import load_labels_split, load_features  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("PASS", name)
    else:
        FAIL.append(name)
        print("FAIL", name, detail)


def test_train_only_geometry():
    gene_gid, gene_score_e, gene_split, comp_dense, n_prom, n_genes = load_labels_split()
    X45, comp44, log1p_count, gc_gene = load_features()
    geom = np.load(MODEL_DIR / "covergate_geometry.npz")
    meta = json.loads((MODEL_DIR / "covergate_geometry.json").read_text(encoding="utf-8"))

    train_idx = np.flatnonzero(gene_split == "train")
    check("geometry_train_idx_matches_split", np.array_equal(geom["train_idx"], train_idx.astype(np.int64)))

    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import LedoitWolf
    X_train = X45[train_idx].astype(np.float64)
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train).astype(np.float64)
    lw = LedoitWolf(store_precision=True).fit(Xs_train)
    check("scaler_mean_matches", np.allclose(geom["mean"], scaler.mean_, atol=1e-12))
    check("scaler_scale_matches", np.allclose(geom["scale"], scaler.scale_, atol=1e-12))
    check("covariance_matches", np.allclose(geom["covariance"], lw.covariance_, atol=1e-12))
    check("precision_matches", np.allclose(geom["precision"], lw.precision_, atol=1e-12))
    train_dist = mahalanobis_sq(Xs_train, lw.location_, lw.precision_)
    thr = quantile_linear(train_dist, PRIMARY_QUANTILE)
    check("primary_threshold_matches", np.isclose(thr, float(geom["threshold"]), atol=1e-9),
          str(thr) + " vs " + str(float(geom["threshold"])))
    check("meta_labels_used_false", meta["fit"]["labels_used"] is False)
    check("meta_heldout_features_used_false", meta["fit"]["heldout_features_used"] is False)


def test_no_label_in_geometry():
    geom = np.load(MODEL_DIR / "covergate_geometry.npz")
    bad = [k for k in geom.files if "score" in k.lower() or "label" in k.lower() or k == "y"]
    check("geometry_has_no_label_arrays", len(bad) == 0, str(bad))
    src = (Path(__file__).resolve().parent / "b7_geometry.py").read_text(encoding="utf-8")
    fit_block = src.split("def main()")[1]
    check("geometry_source_no_gene_score_e_use", "gene_score_e" not in fit_block.split("X_train =")[0])


def test_equal_row_joins():
    import pandas as pd
    gene_gid, gene_score_e, gene_split, comp_dense, n_prom, n_genes = load_labels_split()
    X45, comp44, log1p_count, gc_gene = load_features()
    df = pd.read_csv(REPORT / "predictions" / "pred_v6.8.0-covergate.rc1.tsv", sep="\t")
    df["gene_id"] = df["gene_id"].astype(str)
    ok = True
    detail = ""
    for split_name in ["val", "test"]:
        idx = np.flatnonzero(gene_split == split_name)
        expected = [str(g) for g in gene_gid[idx]]
        got = df[df["split"] == split_name]["gene_id"].tolist()
        if expected != got:
            ok = False
            detail = split_name + " gene_id order mismatch"
        y_expected = gene_score_e[idx]
        y_got = df[df["split"] == split_name]["Score_E"].to_numpy(dtype=float)
        if not np.allclose(y_expected, y_got, atol=1e-12):
            ok = False
            detail = split_name + " Score_E mismatch"
    check("raw_tsv_row_join_exact", ok, detail)


def test_inference_structure():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from b7_inference import CoverGatePredictor
    p = CoverGatePredictor()
    gene_gid, gene_score_e, gene_split, comp_dense, n_prom, n_genes = load_labels_split()
    test_idx = np.flatnonzero(gene_split == "test")
    xs_all = (p.X45.astype(np.float64) - p.mean) / p.scale
    dist = mahalanobis_sq(xs_all, p.location, p.precision)
    order = np.argsort(dist[test_idx], kind="mergesort")
    in_g = str(gene_gid[test_idx[order[0]]])
    out_g = str(gene_gid[test_idx[order[-1]]])
    r_in = p.predict_gene_id(in_g)
    r_out = p.predict_gene_id(out_g)
    check("inference_in_coverage_flag", r_in["coverage_flag"] == "inside_assessed_coverage")
    check("inference_out_coverage_flag", r_out["coverage_flag"] == "outside_assessed_coverage")
    check("inference_no_unreliable_claim",
          "unreliable" not in json.dumps(r_out) and "zero activity" not in json.dumps(r_out))
    check("inference_has_shas", bool(r_in["model_sha256"]) and bool(r_in["geometry_sha256"]))
    check("inference_has_feature_contract", r_in["feature_contract"]["n_features"] == 45)


def test_refusals():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from b7_inference import CoverGatePredictor
    p = CoverGatePredictor()
    check("refuse_single_promoter", p.unsupported_scope("single_promoter")["status"] == "unsupported_scope")
    check("refuse_cell_line", p.unsupported_scope("cell_line_activity")["status"] == "unsupported_scope")
    check("refuse_synthetic", p.unsupported_scope("synthetic_construct")["status"] == "unsupported_scope")
    r = p.predict_gene_id("ENSG00000000000")
    check("missing_context_unknown_gene", r["status"] == "missing_context")
    r2 = p.predict_features(np.zeros(10))
    check("refuse_bad_feature_shape", r2["status"] == "refused_invalid_input")
    r3 = p.predict_features(np.full(45, np.nan))
    check("refuse_nonfinite_features", r3["status"] == "refused_nonfinite_features")


def test_random_mask_baseline():
    gene_gid, gene_score_e, gene_split, comp_dense, n_prom, n_genes = load_labels_split()
    X45, comp44, log1p_count, gc_gene = load_features()
    evalj = json.loads((REPORT / "eval_covergate_b7.json").read_text(encoding="utf-8"))
    t = evalj["by_split"]["test"]
    prim = t["hgb_primary_gate"]
    K = int(prim["n_retained"])
    saved = t["random_mask_maes"]

    import joblib
    hgb = joblib.load(ROOT / "models" / "breakthrough_20261002" / "batch06" / "hgb_baseline.joblib")
    test_idx = np.flatnonzero(gene_split == "test")
    y = gene_score_e[test_idx]
    p = hgb.predict(X45[test_idx])

    rng = np.random.default_rng(RANDOM_MASK_SEED)
    N = len(y)
    maes = []
    all_exact_K = True
    for i in range(RANDOM_MASK_N):
        perm = rng.permutation(N)
        mask = np.zeros(N, dtype=bool)
        mask[perm[:K]] = True
        if int(mask.sum()) != K:
            all_exact_K = False
        maes.append(mae(p[mask], y[mask]))
    check("random_mask_target_count_K", K == int(prim["n_retained"]))
    check("random_mask_all_exact_K", all_exact_K)
    check("random_mask_reproducible", np.allclose(np.asarray(maes), np.asarray(saved), atol=1e-12))
    check("random_mask_n_500", len(saved) == RANDOM_MASK_N)


def main():
    test_train_only_geometry()
    test_no_label_in_geometry()
    test_equal_row_joins()
    test_inference_structure()
    test_refusals()
    test_random_mask_baseline()
    print("-" * 60)
    print("PASS=%d FAIL=%d" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
