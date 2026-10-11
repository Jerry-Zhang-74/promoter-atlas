#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch09 main run: four frozen research fits + affine diagnostic + metrics.

Four research fits only, one deterministic seed, same rows/features/split:
  1) Ridge(alpha=0.1, fit_intercept=True) on continuous raw S (standardized).
  2) Same Ridge on train-reference empirical rank F (standardized).
  3) HistGradientBoostingRegressor(...) on continuous raw S (standardized).
  4) Same HGB on train-reference empirical rank F (standardized).
Plus a separate affine algebra diagnostic (not a fifth arm).
X StandardScaler fits eligible TRAIN only and is reused. Each target is
standardized using eligible TRAIN mean/std and predictions are inverse
transformed back to target units.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b9_common import (ROOT, EXPR, DATA_OUT, MODEL_OUT, REPORT_OUT, RUNS_OUT,
                       CANDIDATE_VERSION, BATCH, SEED, N_BOOTSTRAP, BOOT_LO_PCT,
                       BOOT_HI_PCT, DELTA_MIN, RIDGE, HGB, AFF_A, AFF_B,
                       set_thread_limits, atomic_write_json, reconstruct_s_raw,
                       load_gene_universe, load_features, spearman, r2_score,
                       mae, mse)

set_thread_limits(4)

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import joblib  # noqa: E402


def emp_rank(sorted_train, x):
    """F(x) = (count(train<x) + 0.5*count(train==x)) / n_train, frozen function."""
    n = len(sorted_train)
    lt = np.searchsorted(sorted_train, np.asarray(x, dtype=np.float64), side="left")
    eq = np.searchsorted(sorted_train, np.asarray(x, dtype=np.float64), side="right") - lt
    return (lt.astype(np.float64) + 0.5 * eq.astype(np.float64)) / float(n)


def spearman_scipy(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(sps.spearmanr(x, y).statistic)


def main():
    t0 = time.time()
    for d in (DATA_OUT, MODEL_OUT, REPORT_OUT, RUNS_OUT):
        d.mkdir(parents=True, exist_ok=True)

    # ---- load universe / split / features / labels ----
    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    X_all = load_features()
    full_s = reconstruct_s_raw(EXPR)

    eligible_mask = np.array([gid in full_s for gid in gene_gid])
    n_eligible = int(eligible_mask.sum())
    assert n_eligible == 18769

    gene_id_e = gene_gid[eligible_mask]
    split_e = gene_split[eligible_mask]
    comp_e = comp_dense[eligible_mask]
    n_prom_e = n_prom[eligible_mask]
    X_e = X_all[eligible_mask]
    s_raw = np.array([full_s[g] for g in gene_id_e], dtype=np.float64)
    assert np.isfinite(s_raw).all()

    # split masks (eligible rows)
    tr = split_e == "train"
    va = split_e == "val"
    te = split_e == "test"
    n_tr, n_va, n_te = int(tr.sum()), int(va.sum()), int(te.sum())

    # ---- X scaler: fit on eligible TRAIN only ----
    scaler = StandardScaler()
    scaler.fit(X_e[tr])
    Xs = scaler.transform(X_e)

    # ---- train-reference empirical rank (frozen on train only) ----
    sorted_train = np.sort(s_raw[tr])
    n_train = len(sorted_train)
    F_all = emp_rank(sorted_train, s_raw)
    F_tr, F_va, F_te = F_all[tr], F_all[va], F_all[te]

    # out-of-range / induced tie audit
    below = int((s_raw[~tr] < sorted_train[0]).sum())
    above = int((s_raw[~tr] > sorted_train[-1]).sum())
    out_of_range = {"val_test_below_train_min": below, "val_test_above_train_max": above}
    # induced ties: non-train values sharing the same F value
    f_non_tr = F_all[~tr]
    _, counts = np.unique(f_non_tr, return_counts=True)
    tied_f_values = int((counts > 1).sum())
    tied_rows = int((counts[counts > 1]).sum())
    induced_ties = {"n_distinct_F_non_train": int(np.unique(f_non_tr).size),
                    "n_F_values_with_ties": tied_f_values,
                    "n_non_train_rows_in_tied_F": tied_rows,
                    "n_non_train_rows": int((~tr).sum())}

    # ---- target standardization (train mean/std, population std) ----
    tr_mean_raw = float(s_raw[tr].mean())
    tr_std_raw = float(s_raw[tr].std(ddof=0))
    tr_mean_rank = float(F_tr.mean())
    tr_std_rank = float(F_tr.std(ddof=0))

    z_raw = (s_raw - tr_mean_raw) / tr_std_raw
    z_rank = (F_all - tr_mean_rank) / tr_std_rank

    Xs_tr, Xs_va, Xs_te = Xs[tr], Xs[va], Xs[te]

    # ---- four frozen research fits ----
    models = {}
    preds = {}  # inverse-transformed predictions in target units

    m = Ridge(**RIDGE)
    m.fit(Xs_tr, z_raw[tr])
    models["ridge_raw"] = m
    preds["ridge_raw"] = m.predict(Xs) * tr_std_raw + tr_mean_raw

    m = Ridge(**RIDGE)
    m.fit(Xs_tr, z_rank[tr])
    models["ridge_rank"] = m
    preds["ridge_rank"] = m.predict(Xs) * tr_std_rank + tr_mean_rank

    m = HistGradientBoostingRegressor(**HGB)
    m.fit(Xs_tr, z_raw[tr])
    models["hgb_raw"] = m
    preds["hgb_raw"] = m.predict(Xs) * tr_std_raw + tr_mean_raw

    m = HistGradientBoostingRegressor(**HGB)
    m.fit(Xs_tr, z_rank[tr])
    models["hgb_rank"] = m
    preds["hgb_rank"] = m.predict(Xs) * tr_std_rank + tr_mean_rank

    # ---- affine diagnostic (algebra test; NOT a fifth arm) ----
    y_aff1 = s_raw[tr]
    y_aff2 = AFF_A + AFF_B * y_aff1
    m1 = Ridge(**RIDGE).fit(Xs_tr, y_aff1)
    m2 = Ridge(**RIDGE).fit(Xs_tr, y_aff2)
    p1 = m1.predict(Xs)
    p2 = m2.predict(Xs)
    aff_max_err = float(np.max(np.abs(p2 - (AFF_A + AFF_B * p1))))

    # affine metrics on TEST rows (paired)
    y1_te = s_raw[te]
    y2_te = AFF_A + AFF_B * y1_te
    p1_te = p1[te]
    p2_te = p2[te]
    aff_metrics = {
        "max_abs_affine_err_all_rows": aff_max_err,
        "spearman_y1_vs_p1": spearman(y1_te, p1_te),
        "spearman_y2_vs_p2": spearman(y2_te, p2_te),
        "spearman_invariant": abs(spearman(y1_te, p1_te) - spearman(y2_te, p2_te)) < 1e-12,
        "r2_y1": r2_score(y1_te, p1_te),
        "r2_y2": r2_score(y2_te, p2_te),
        "r2_invariant": abs(r2_score(y1_te, p1_te) - r2_score(y2_te, p2_te)) < 1e-12,
        "mae_y1": mae(y1_te, p1_te),
        "mae_y2": mae(y2_te, p2_te),
        "mae_ratio_y2_over_y1": mae(y2_te, p2_te) / mae(y1_te, p1_te),
        "expected_mae_ratio": AFF_B,
        "mse_y1": mse(y1_te, p1_te),
        "mse_y2": mse(y2_te, p2_te),
        "mse_ratio_y2_over_y1": mse(y2_te, p2_te) / mse(y1_te, p1_te),
        "expected_mse_ratio": AFF_B ** 2,
    }

    # ---- point metrics on test ----
    def arm_metrics(pred, y_true):
        return {
            "spearman_vs_raw_truth": spearman(pred[te], s_raw[te]),
            "mae": mae(y_true[te], pred[te]),
            "mse": mse(y_true[te], pred[te]),
            "r2": r2_score(y_true[te], pred[te]),
        }

    metrics_arms = {
        "ridge_raw": arm_metrics(preds["ridge_raw"], s_raw),
        "ridge_rank": arm_metrics(preds["ridge_rank"], F_all),
        "hgb_raw": arm_metrics(preds["hgb_raw"], s_raw),
        "hgb_rank": arm_metrics(preds["hgb_rank"], F_all),
    }

    # ---- descriptive VAL metrics (no selection, no fitting on val) ----
    val_descriptive = {
        "spearman_vs_raw_truth": {
            "ridge_raw": spearman(preds["ridge_raw"][va], s_raw[va]),
            "ridge_rank": spearman(preds["ridge_rank"][va], s_raw[va]),
            "hgb_raw": spearman(preds["hgb_raw"][va], s_raw[va]),
            "hgb_rank": spearman(preds["hgb_rank"][va], s_raw[va]),
        },
        "within_target": {
            "ridge_raw": {"mae": mae(s_raw[va], preds["ridge_raw"][va]),
                          "mse": mse(s_raw[va], preds["ridge_raw"][va]),
                          "r2": r2_score(s_raw[va], preds["ridge_raw"][va])},
            "ridge_rank": {"mae": mae(F_va, preds["ridge_rank"][va]),
                           "mse": mse(F_va, preds["ridge_rank"][va]),
                           "r2": r2_score(F_va, preds["ridge_rank"][va])},
            "hgb_raw": {"mae": mae(s_raw[va], preds["hgb_raw"][va]),
                        "mse": mse(s_raw[va], preds["hgb_raw"][va]),
                        "r2": r2_score(s_raw[va], preds["hgb_raw"][va])},
            "hgb_rank": {"mae": mae(F_va, preds["hgb_rank"][va]),
                         "mse": mse(F_va, preds["hgb_rank"][va]),
                         "r2": r2_score(F_va, preds["hgb_rank"][va])},
        },
        "note": "descriptive only; no scaler/CDF/hyperparameter/stopping/target mapping is fit or selected on val",
    }

    # ---- train mean / median baselines (per target) ----
    def baseline_metrics(y_true_train, y_true_test):
        out = {}
        for name, const in [("train_mean", float(y_true_train.mean())),
                            ("train_median", float(np.median(y_true_train)))]:
            p = np.full(len(y_true_test), const)
            out[name] = {"mae": mae(y_true_test, p), "mse": mse(y_true_test, p),
                         "r2": r2_score(y_true_test, p)}
        return out

    baselines = {
        "raw_target": baseline_metrics(s_raw[tr], s_raw[te]),
        "rank_target": baseline_metrics(F_tr, F_te),
    }

    # ---- paired component bootstrap for deltas ----
    comp_te = comp_e[te]
    te_idx = np.flatnonzero(te)
    uniq_comp = np.unique(comp_te)
    rng = np.random.default_rng(SEED)
    s_raw_te = s_raw[te]
    sp_rank_full = spearman(preds["ridge_rank"][te], s_raw_te)
    sp_raw_full = spearman(preds["ridge_raw"][te], s_raw_te)

    comp_list = {c: np.flatnonzero(comp_te == c) for c in uniq_comp}  # positions within test
    deltas = {"ridge": np.empty(N_BOOTSTRAP), "hgb": np.empty(N_BOOTSTRAP)}
    for b in range(N_BOOTSTRAP):
        sample = rng.choice(uniq_comp, size=len(uniq_comp), replace=True)
        idx = np.concatenate([comp_list[c] for c in sample])
        truth_b = s_raw_te[idx]
        d_ridge = spearman(preds["ridge_rank"][te][idx], truth_b) - spearman(preds["ridge_raw"][te][idx], truth_b)
        d_hgb = spearman(preds["hgb_rank"][te][idx], truth_b) - spearman(preds["hgb_raw"][te][idx], truth_b)
        deltas["ridge"][b] = d_ridge
        deltas["hgb"][b] = d_hgb

    def boot_summary(arr, point):
        lo = float(np.percentile(arr, BOOT_LO_PCT))
        hi = float(np.percentile(arr, BOOT_HI_PCT))
        return {
            "point_estimate": float(point),
            "bootstrap_mean": float(arr.mean()),
            "lo_pct": BOOT_LO_PCT,
            "hi_pct": BOOT_HI_PCT,
            "lo": lo,
            "hi": hi,
            "n_resamples": N_BOOTSTRAP,
            "n_bootstrap_nan": int(np.isnan(arr).sum()),
        }

    delta_ridge_point = spearman(preds["ridge_rank"][te], s_raw_te) - spearman(preds["ridge_raw"][te], s_raw_te)
    delta_hgb_point = spearman(preds["hgb_rank"][te], s_raw_te) - spearman(preds["hgb_raw"][te], s_raw_te)
    primary = {
        "convention": ("Two primary comparisons; each uses a 97.5%% two-sided percentile "
                       "interval ([%.2f, %.2f] percentiles of the paired component bootstrap "
                       "delta distribution) as a conservative two-comparison adjustment." % (BOOT_LO_PCT, BOOT_HI_PCT)),
        "acceptance_rule": ("scoped target-representation improvement requires delta >= %.2f AND "
                            "adjusted lower bound > 0" % DELTA_MIN),
        "ridge": boot_summary(deltas["ridge"], delta_ridge_point),
        "hgb": boot_summary(deltas["hgb"], delta_hgb_point),
        "ridge_meets_rule": bool(delta_ridge_point >= DELTA_MIN and np.percentile(deltas["ridge"], BOOT_LO_PCT) > 0),
        "hgb_meets_rule": bool(delta_hgb_point >= DELTA_MIN and np.percentile(deltas["hgb"], BOOT_LO_PCT) > 0),
    }

    # ---- write prediction TSV (all eligible rows) ----
    pred_df = pd.DataFrame({
        "gene_id": gene_id_e,
        "split": split_e,
        "component_id": comp_e,
        "n_promoters": n_prom_e,
        "S_raw_full": s_raw,
        "rank_target_F": F_all,
        "ridge_raw_pred": preds["ridge_raw"],
        "ridge_rank_pred": preds["ridge_rank"],
        "hgb_raw_pred": preds["hgb_raw"],
        "hgb_rank_pred": preds["hgb_rank"],
    })
    pred_df.to_csv(DATA_OUT / "predictions.tsv", sep="\t", index=False, float_format="%.17g")

    # ---- serialize transforms + models ----
    joblib.dump({
        "ridge_raw": models["ridge_raw"],
        "ridge_rank": models["ridge_rank"],
        "hgb_raw": models["hgb_raw"],
        "hgb_rank": models["hgb_rank"],
    }, MODEL_OUT / "models.joblib")
    joblib.dump(scaler, MODEL_OUT / "X_scaler.joblib")
    np.savez_compressed(MODEL_OUT / "rank_reference.npz", sorted_train=sorted_train)
    transforms = {
        "X_scaler": {"mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist()},
        "target_raw": {"train_mean": tr_mean_raw, "train_std": tr_std_raw},
        "target_rank": {"train_mean": tr_mean_rank, "train_std": tr_std_rank},
        "rank_reference": {"n_train": n_train, "train_min": float(sorted_train[0]),
                           "train_max": float(sorted_train[-1])},
    }
    atomic_write_json(MODEL_OUT / "transforms.json", transforms)

    # ---- metrics.json ----
    result = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "n_eligible": n_eligible,
        "n_train": n_tr, "n_val": n_va, "n_test": n_te,
        "rank_target_audit": {
            "n_train": n_train,
            "out_of_range": out_of_range,
            "induced_ties": induced_ties,
            "clip_behavior": "values below train min -> 0; above train max -> 1",
        },
        "target_standardization": {
            "raw_train_mean": tr_mean_raw, "raw_train_std": tr_std_raw,
            "rank_train_mean": tr_mean_rank, "rank_train_std": tr_std_rank,
        },
        "arms": metrics_arms,
        "val_descriptive": val_descriptive,
        "baselines": baselines,
        "primary_target_representation": primary,
        "ridge_vs_hgb_descriptive": {
            "test_spearman_vs_raw_truth": {
                "ridge_raw": metrics_arms["ridge_raw"]["spearman_vs_raw_truth"],
                "ridge_rank": metrics_arms["ridge_rank"]["spearman_vs_raw_truth"],
                "hgb_raw": metrics_arms["hgb_raw"]["spearman_vs_raw_truth"],
                "hgb_rank": metrics_arms["hgb_rank"]["spearman_vs_raw_truth"],
            },
            "note": "descriptive only; not an acceptance comparison and not compared to batch06 E or v6.3",
        },
        "affine_diagnostic": aff_metrics,
        "affine_diagnostic_pass": bool(aff_max_err < 1e-8 and aff_metrics["r2_invariant"] and aff_metrics["spearman_invariant"]),
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(REPORT_OUT / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
