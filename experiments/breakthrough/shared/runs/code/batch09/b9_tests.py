#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch09 independent tests (scientific failure modes, not implementation mirroring).

Each test re-derives a scientific invariant from the saved artifacts or the raw
source. They do NOT re-run the research fits (those are already frozen); they
check provenance, transforms, the rank formula, affine algebra, saved-model
replay and metric/artifact consistency.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b9_common import (ROOT, EXPR, DATA_OUT, MODEL_OUT, REPORT_OUT,
                       CANDIDATE_VERSION, BATCH, reconstruct_s_raw,
                       load_gene_universe, load_features, spearman, r2_score,
                       mae, mse, atomic_write_json)

from sklearn.linear_model import Ridge  # noqa: E402
import joblib  # noqa: E402


def emp_rank(sorted_train, x):
    n = len(sorted_train)
    x = np.asarray(x, dtype=np.float64)
    lt = np.searchsorted(sorted_train, x, side="left")
    eq = np.searchsorted(sorted_train, x, side="right") - lt
    return (lt.astype(np.float64) + 0.5 * eq.astype(np.float64)) / float(n)


def check(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": detail}


def t_join_identity(pred, full_s, gene_gid):
    gene_id_e = pred["gene_id"].values
    split_e = pred["split"].values
    feat_ids = set(gene_gid)
    # eligible = S-defined AND feature-covered
    eligible_ids = set(full_s) & feat_ids
    ok = set(gene_id_e) == eligible_ids and len(gene_id_e) == 18769
    # each eligible gene appears exactly once with one split
    uniq = pred["gene_id"].nunique() == len(pred)
    return check("join_identity", ok and uniq,
                 "eligible set == full_s ∩ feature universe (18769); each gene unique: %s" % uniq)


def t_excluded_count(pred, full_s, gene_gid):
    feat_ids = set(gene_gid)
    n_excl = len(set(full_s) - feat_ids)
    return check("excluded_count_178", n_excl == 178, "excluded canonical S-defined = %d" % n_excl)


def t_no_component_crossing(pred):
    g = pred.groupby("component_id")["split"].nunique()
    bad = int((g > 1).sum())
    return check("no_component_crossing", bad == 0,
                 "components with >1 split = %d" % bad)


def t_train_only_scaler(pred, X_e, gene_split):
    tr = pred["split"].values == "train"
    mu_train = X_e[tr].mean(axis=0)
    sd_train = X_e[tr].std(axis=0)
    sd_train[sd_train == 0] = 1.0
    sc = joblib.load(MODEL_OUT / "X_scaler.joblib")
    ok_mean = np.allclose(sc.mean_, mu_train)
    ok_scale = np.allclose(sc.scale_, sd_train)
    # sanity: scaler mean != full-data mean (would catch val/test leakage)
    full_mean = X_e.mean(axis=0)
    leaked = np.allclose(sc.mean_, full_mean)
    return check("train_only_scaler", bool(ok_mean and ok_scale and not leaked),
                 "scaler mean==train mean: %s; scale==train std: %s; no full-data leakage: %s"
                 % (ok_mean, ok_scale, not leaked))


def t_train_only_target_standardization(pred):
    s = pred["S_raw_full"].values
    F = pred["rank_target_F"].values
    tr = pred["split"].values == "train"
    tr_mean_raw = float(s[tr].mean()); tr_std_raw = float(s[tr].std(ddof=0))
    tr_mean_rank = float(F[tr].mean()); tr_std_rank = float(F[tr].std(ddof=0))
    tf = json.load(open(MODEL_OUT / "transforms.json", encoding="utf-8"))
    ok = (abs(tf["target_raw"]["train_mean"] - tr_mean_raw) < 1e-12 and
          abs(tf["target_raw"]["train_std"] - tr_std_raw) < 1e-12 and
          abs(tf["target_rank"]["train_mean"] - tr_mean_rank) < 1e-12 and
          abs(tf["target_rank"]["train_std"] - tr_std_rank) < 1e-12)
    return check("train_only_target_standardization", ok,
                 "saved target train mean/std match recomputed train-only values")


def t_rank_formula_synthetic():
    # small synthetic reference; verify midpoint formula and tail clipping
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    cases = [0.0, 1.0, 1.5, 2.0, 4.0, 9.0]
    got = emp_rank(ref, cases)
    # F(1.0) = (0 + 0.5)/4 = 0.125; F(1.5)=1/4=0.25; F(2)=1.5/4=0.375;
    # F(4)=3.5/4=0.875; F(0)=0; F(9)=1
    exp = np.array([0.0, 0.125, 0.25, 0.375, 0.875, 1.0])
    ok = np.allclose(got, exp)
    return check("rank_formula_synthetic", ok,
                 "midpoint + tail mapping verified on synthetic reference; got %s" % got.tolist())


def t_rank_ties_and_range(pred):
    s = pred["S_raw_full"].values
    F = pred["rank_target_F"].values
    tr = pred["split"].values == "train"
    sorted_train = np.sort(s[tr])
    F_re = emp_rank(sorted_train, s)
    ok_formula = np.allclose(F, F_re, atol=1e-12)
    below = int((s[~tr] < sorted_train[0]).sum())
    above = int((s[~tr] > sorted_train[-1]).sum())
    # induced ties among non-train rows
    fn = F[~tr]
    _, cnt = np.unique(fn, return_counts=True)
    tied = int((cnt > 1).sum())
    return check("rank_ties_and_range", bool(ok_formula and below == 0 and above == 0 and tied > 0),
                 "frozen F reproduced; out-of-range below=%d above=%d; tied F-values=%d" % (below, above, tied))


def t_target_standardization_inversion(pred):
    s = pred["S_raw_full"].values
    tr = pred["split"].values == "train"
    mu = s[tr].mean(); sd = s[tr].std(ddof=0)
    z = (s - mu) / sd
    inv = z * sd + mu
    return check("target_standardization_inversion", np.allclose(inv, s, atol=1e-12),
                 "raw target standardization then inverse recovers raw S")


def t_missing_vs_legitimate_zero(pred, full_s, gene_gid):
    s = pred["S_raw_full"].values
    ok_no_nan = np.isfinite(s).all()
    # S is -MAD <= 0; all eligible defined values should be strictly negative
    # (a zero would be a legitimate MAD=0 case, not missingness, but is absent here)
    n_zero = int((np.abs(s) < 1e-15).sum())
    # no eligible gene should have undefined S (missing) because join is by set
    ok_all_negative = (s < 0).all()
    detail = "no NaN: %s; zero values: %d; all strictly negative: %s (missing S excluded by join)" % (
        ok_no_nan, n_zero, ok_all_negative)
    return check("missing_vs_legitimate_zero", bool(ok_no_nan and ok_all_negative and n_zero == 0), detail)


def t_saved_model_replay(pred, full_s, gene_gid, X_e):
    models = joblib.load(MODEL_OUT / "models.joblib")
    sc = joblib.load(MODEL_OUT / "X_scaler.joblib")
    ref = np.load(MODEL_OUT / "rank_reference.npz")["sorted_train"]
    tf = json.load(open(MODEL_OUT / "transforms.json", encoding="utf-8"))
    Xs = sc.transform(X_e)
    eligible = np.array([g in full_s for g in gene_gid])
    Xs_e = Xs[eligible]
    s_raw = np.array([full_s[g] for g in gene_gid[eligible]])
    F = emp_rank(ref, s_raw)
    p_ridge_raw = models["ridge_raw"].predict(Xs_e) * tf["target_raw"]["train_std"] + tf["target_raw"]["train_mean"]
    p_ridge_rank = models["ridge_rank"].predict(Xs_e) * tf["target_rank"]["train_std"] + tf["target_rank"]["train_mean"]
    p_hgb_raw = models["hgb_raw"].predict(Xs_e) * tf["target_raw"]["train_std"] + tf["target_raw"]["train_mean"]
    p_hgb_rank = models["hgb_rank"].predict(Xs_e) * tf["target_rank"]["train_std"] + tf["target_rank"]["train_mean"]
    ok = (np.allclose(p_ridge_raw, pred["ridge_raw_pred"].values, atol=1e-8) and
          np.allclose(p_ridge_rank, pred["ridge_rank_pred"].values, atol=1e-8) and
          np.allclose(p_hgb_raw, pred["hgb_raw_pred"].values, atol=1e-8) and
          np.allclose(p_hgb_rank, pred["hgb_rank_pred"].values, atol=1e-8))
    return check("saved_model_replay", bool(ok),
                 "replayed all four arm predictions from serialized models/scaler/rank ref match predictions.tsv")


def t_affine_diagnostic_deterministic(pred, full_s, gene_gid, gene_split, X_e):
    from b9_common import AFF_A, AFF_B, RIDGE
    eligible = np.array([g in full_s for g in gene_gid])
    X_e = X_e[eligible]
    s_raw = np.array([full_s[g] for g in gene_gid[eligible]])
    tr = (gene_split[eligible] == "train")
    sc = joblib.load(MODEL_OUT / "X_scaler.joblib")
    Xs = sc.transform(X_e)
    y1 = s_raw[tr]
    y2 = AFF_A + AFF_B * y1
    m1 = Ridge(**RIDGE).fit(Xs[tr], y1)
    m2 = Ridge(**RIDGE).fit(Xs[tr], y2)
    p1 = m1.predict(Xs); p2 = m2.predict(Xs)
    err = float(np.max(np.abs(p2 - (AFF_A + AFF_B * p1))))
    return check("affine_diagnostic_deterministic", err < 1e-8,
                 "Ridge affine-related predictions max err = %.3e (tol 1e-8)" % err)


def t_metric_artifact_consistency(pred):
    metrics = json.load(open(REPORT_OUT / "metrics.json", encoding="utf-8"))
    te = pred["split"].values == "test"
    s = pred["S_raw_full"].values
    sp = {}
    for arm in ("ridge_raw", "ridge_rank", "hgb_raw", "hgb_rank"):
        sp[arm] = spearman(pred[arm + "_pred"].values[te], s[te])
    ok = all(abs(sp[k] - metrics["arms"][k]["spearman_vs_raw_truth"]) < 1e-12 for k in sp)
    # primary delta point estimates
    d_r = sp["ridge_rank"] - sp["ridge_raw"]
    d_h = sp["hgb_rank"] - sp["hgb_raw"]
    ok2 = (abs(d_r - metrics["primary_target_representation"]["ridge"]["point_estimate"]) < 1e-12 and
           abs(d_h - metrics["primary_target_representation"]["hgb"]["point_estimate"]) < 1e-12)
    return check("metric_artifact_consistency", bool(ok and ok2),
                 "test Spearman and primary deltas recomputed from predictions.tsv match metrics.json")


def main():
    t0 = time.time()
    pred = pd.read_csv(DATA_OUT / "predictions.tsv", sep="\t", low_memory=False)
    full_s = reconstruct_s_raw(EXPR)
    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    X_all = load_features()

    results = [
        t_join_identity(pred, full_s, gene_gid),
        t_excluded_count(pred, full_s, gene_gid),
        t_no_component_crossing(pred),
        t_train_only_scaler(pred, X_all[np.array([g in full_s for g in gene_gid])], gene_split),
        t_train_only_target_standardization(pred),
        t_rank_formula_synthetic(),
        t_rank_ties_and_range(pred),
        t_target_standardization_inversion(pred),
        t_missing_vs_legitimate_zero(pred, full_s, gene_gid),
        t_saved_model_replay(pred, full_s, gene_gid, X_all),
        t_affine_diagnostic_deterministic(pred, full_s, gene_gid, gene_split, X_all),
        t_metric_artifact_consistency(pred),
    ]
    summary = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tests": results,
        "n_pass": int(sum(1 for r in results if r["pass"])),
        "n_total": len(results),
        "all_pass": all(r["pass"] for r in results),
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(REPORT_OUT / "tests_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
