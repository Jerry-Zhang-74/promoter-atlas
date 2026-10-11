#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch10 main run: two frozen scientific fits (augmented Ridge + HGB) + metrics.

Exactly two new scientific fits on the 1069-feature matrix (45 baseline + 1024
new positional sequence features) against continuous raw S:
  1) Ridge(alpha=0.1, fit_intercept=True)
  2) HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
     max_leaf_nodes=15, l2_regularization=0.1, early_stopping=False,
     random_state=20261011)
X StandardScaler fits eligible TRAIN only; target standardized on eligible TRAIN
mean/std and predictions inverse-transformed to raw S units. Batch09 ridge_raw /
hgb_raw saved predictions are reused as matched baselines (no refit). Two
co-primary paired component bootstrap comparisons: augmented-minus-baseline TEST
Spearman within Ridge and within HGB.

Entry point FAILS if its target attempt dir already contains fit artifacts.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from b10_common import (ROOT, EXPR, DATA_OUT, MODEL_OUT, REPORT_OUT, B9_PRED,
                        CANDIDATE_VERSION, BATCH, SEED, N_BOOTSTRAP, BOOT_LO_PCT,
                        BOOT_HI_PCT, DELTA_MIN, RIDGE, HGB,
                        FEATURE_NAMES_ALL, sha256_file, atomic_write_json,
                        reconstruct_s_raw, load_gene_universe, load_features45,
                        spearman, r2_score, mae, mse, thread_caps_record,
                        N_EXPECTED_ELIGIBLE)  # noqa: E402

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import joblib  # noqa: E402


def append_ledger(record):
    p = DATA_OUT / "attempts.jsonl"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def snapshot_code(attempt_dir):
    src_dir = Path(__file__).resolve().parent
    dst = attempt_dir / "code"
    dst.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for f in sorted(src_dir.glob("b10_*.py")):
        shutil.copy2(f, dst / f.name)
        hashes[f.name] = sha256_file(f)
    return hashes


def load_augmented_x(extract_attempt):
    xdir = DATA_OUT / ("attempt_%03d_extract" % extract_attempt)
    X1024 = np.load(xdir / "features1024.npz")["X1024"].astype(np.float64)
    X45 = load_features45()
    assert X1024.shape == (X45.shape[0], 1024)
    X = np.concatenate([X45, X1024], axis=1)
    assert X.shape[1] == 1069
    assert len(FEATURE_NAMES_ALL) == X.shape[1]
    return X, xdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--extract-attempt", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    attempt_dir = MODEL_OUT / ("attempt_%03d_fit" % args.attempt)
    # ---- non-overwrite guard ----
    if (attempt_dir / "models.joblib").exists():
        raise SystemExit("REFUSE: fit attempt output already exists: %s (no overwrite)"
                         % (attempt_dir / "models.joblib"))
    attempt_dir.mkdir(parents=True, exist_ok=True)

    code_hashes = snapshot_code(attempt_dir)
    atomic_write_json(attempt_dir / "code_hashes.json", code_hashes)

    with threadpool_limits(limits=4):
        # ---- load universe / features / labels ----
        gene_gid, gene_split, comp_dense, n_prom, starts, gene_id_perprom = load_gene_universe()
        X_all, xdir = load_augmented_x(args.extract_attempt)
        full_s = reconstruct_s_raw(EXPR)

        eligible_mask = np.array([gid in full_s for gid in gene_gid])
        n_eligible = int(eligible_mask.sum())
        assert n_eligible == N_EXPECTED_ELIGIBLE

        gene_id_e = gene_gid[eligible_mask]
        split_e = gene_split[eligible_mask]
        comp_e = comp_dense[eligible_mask]
        n_prom_e = n_prom[eligible_mask]
        X_e = X_all[eligible_mask]
        s_raw = np.array([full_s[g] for g in gene_id_e], dtype=np.float64)
        assert np.isfinite(s_raw).all()

        tr = split_e == "train"
        va = split_e == "val"
        te = split_e == "test"
        n_tr, n_va, n_te = int(tr.sum()), int(va.sum()), int(te.sum())
        assert (n_tr, n_va, n_te) == (15004, 1898, 1867)

        # ---- X scaler: fit on eligible TRAIN only ----
        scaler = StandardScaler()
        scaler.fit(X_e[tr])
        Xs = scaler.transform(X_e)

        # ---- target standardization (train mean/std, population std) ----
        tr_mean_raw = float(s_raw[tr].mean())
        tr_std_raw = float(s_raw[tr].std(ddof=0))
        z_raw = (s_raw - tr_mean_raw) / tr_std_raw
        Xs_tr = Xs[tr]

        # ---- two frozen scientific fits ----
        models = {}
        preds = {}

        m = Ridge(**RIDGE)
        m.fit(Xs_tr, z_raw[tr])
        models["ridge_aug"] = m
        preds["ridge_aug"] = m.predict(Xs) * tr_std_raw + tr_mean_raw

        m = HistGradientBoostingRegressor(**HGB)
        m.fit(Xs_tr, z_raw[tr])
        models["hgb_aug"] = m
        preds["hgb_aug"] = m.predict(Xs) * tr_std_raw + tr_mean_raw

    # ---- load batch09 baselines (no refit) and align by gene id ----
    b9 = pd.read_csv(B9_PRED, sep="\t", low_memory=False)
    b9_map = {row.gene_id: row for row in b9.itertuples(index=False)}
    base_ridge = np.array([b9_map[g].ridge_raw_pred for g in gene_id_e], dtype=np.float64)
    base_hgb = np.array([b9_map[g].hgb_raw_pred for g in gene_id_e], dtype=np.float64)
    b9_s = np.array([b9_map[g].S_raw_full for g in gene_id_e], dtype=np.float64)
    assert np.allclose(b9_s, s_raw, atol=1e-12), "batch09 truth != batch10 reconstructed S"
    assert np.array_equal(np.array([b9_map[g].split for g in gene_id_e]), split_e)
    assert np.array_equal(np.array([b9_map[g].component_id for g in gene_id_e]), comp_e)

    # ---- point metrics on TEST (vs full raw truth) ----
    def arm_metrics(pred, y_true):
        return {
            "spearman_vs_raw_truth": spearman(pred[te], y_true[te]),
            "mae": mae(y_true[te], pred[te]),
            "mse": mse(y_true[te], pred[te]),
            "r2": r2_score(y_true[te], pred[te]),
        }

    aug_metrics = {
        "ridge_aug": arm_metrics(preds["ridge_aug"], s_raw),
        "hgb_aug": arm_metrics(preds["hgb_aug"], s_raw),
    }
    base_metrics = {
        "ridge_base": arm_metrics(base_ridge, s_raw),
        "hgb_base": arm_metrics(base_hgb, s_raw),
    }

    # ---- descriptive VAL metrics ----
    val_descriptive = {
        "spearman_vs_raw_truth": {
            "ridge_aug": spearman(preds["ridge_aug"][va], s_raw[va]),
            "hgb_aug": spearman(preds["hgb_aug"][va], s_raw[va]),
            "ridge_base": spearman(base_ridge[va], s_raw[va]),
            "hgb_base": spearman(base_hgb[va], s_raw[va]),
        },
        "within_raw_target": {
            "ridge_aug": {"mae": mae(s_raw[va], preds["ridge_aug"][va]),
                          "mse": mse(s_raw[va], preds["ridge_aug"][va]),
                          "r2": r2_score(s_raw[va], preds["ridge_aug"][va])},
            "hgb_aug": {"mae": mae(s_raw[va], preds["hgb_aug"][va]),
                        "mse": mse(s_raw[va], preds["hgb_aug"][va]),
                        "r2": r2_score(s_raw[va], preds["hgb_aug"][va])},
        },
        "note": "descriptive only; no scaler/hyperparameter/stopping/target mapping fit or selected on val",
    }

    # ---- train mean / median baselines (raw units) ----
    def baseline_metrics(y_tr, y_te):
        out = {}
        for name, const in [("train_mean", float(np.mean(y_tr))),
                            ("train_median", float(np.median(y_tr)))]:
            p = np.full(len(y_te), const)
            out[name] = {"mae": mae(y_te, p), "mse": mse(y_te, p), "r2": r2_score(y_te, p)}
        return out

    const_baselines = baseline_metrics(s_raw[tr], s_raw[te])

    # ---- raw-unit deltas (aug - base) on test ----
    def raw_deltas(p_aug, p_base, y):
        return {
            "mae_delta": mae(y, p_aug) - mae(y, p_base),
            "mse_delta": mse(y, p_aug) - mse(y, p_base),
            "r2_delta": r2_score(y, p_aug) - r2_score(y, p_base),
        }

    raw_deltas_out = {
        "ridge": raw_deltas(preds["ridge_aug"][te], base_ridge[te], s_raw[te]),
        "hgb": raw_deltas(preds["hgb_aug"][te], base_hgb[te], s_raw[te]),
    }

    # ---- paired component bootstrap: delta = aug - base Spearman ----
    comp_te = comp_e[te]
    te_idx = np.flatnonzero(te)
    uniq_comp = np.unique(comp_te)
    rng = np.random.default_rng(SEED)
    s_raw_te = s_raw[te]
    comp_list = {int(c): np.flatnonzero(comp_te == c) for c in uniq_comp}

    pairs = {
        "ridge": (preds["ridge_aug"][te], base_ridge[te]),
        "hgb": (preds["hgb_aug"][te], base_hgb[te]),
    }
    deltas = {k: np.empty(N_BOOTSTRAP) for k in pairs}
    for b in range(N_BOOTSTRAP):
        sample = rng.choice(uniq_comp, size=len(uniq_comp), replace=True)
        idx = np.concatenate([comp_list[int(c)] for c in sample])
        truth_b = s_raw_te[idx]
        for k, (paug, pbase) in pairs.items():
            deltas[k][b] = spearman(paug[idx], truth_b) - spearman(pbase[idx], truth_b)

    def boot_summary(arr, point):
        return {
            "point_estimate": float(point),
            "bootstrap_mean": float(arr.mean()),
            "lo_pct": BOOT_LO_PCT,
            "hi_pct": BOOT_HI_PCT,
            "lo": float(np.percentile(arr, BOOT_LO_PCT)),
            "hi": float(np.percentile(arr, BOOT_HI_PCT)),
            "n_resamples": N_BOOTSTRAP,
            "n_bootstrap_nan": int(np.isnan(arr).sum()),
        }

    delta_ridge_point = spearman(preds["ridge_aug"][te], s_raw_te) - spearman(base_ridge[te], s_raw_te)
    delta_hgb_point = spearman(preds["hgb_aug"][te], s_raw_te) - spearman(base_hgb[te], s_raw_te)
    primary = {
        "convention": ("Two primary comparisons; each uses a 97.5%% two-sided percentile "
                       "interval ([%.2f, %.2f] percentiles of the paired component bootstrap "
                       "delta distribution)." % (BOOT_LO_PCT, BOOT_HI_PCT)),
        "acceptance_rule": "scoped feature improvement requires delta >= %.2f AND adjusted lower bound > 0" % DELTA_MIN,
        "ridge": boot_summary(deltas["ridge"], delta_ridge_point),
        "hgb": boot_summary(deltas["hgb"], delta_hgb_point),
        "ridge_meets_rule": bool(delta_ridge_point >= DELTA_MIN and np.percentile(deltas["ridge"], BOOT_LO_PCT) > 0),
        "hgb_meets_rule": bool(delta_hgb_point >= DELTA_MIN and np.percentile(deltas["hgb"], BOOT_LO_PCT) > 0),
    }

    # ---- write per-gene predictions (universe-eligible order) ----
    pred_df = pd.DataFrame({
        "gene_id": gene_id_e,
        "split": split_e,
        "component_id": comp_e,
        "n_promoters": n_prom_e,
        "S_raw_full": s_raw,
        "ridge_aug_pred": preds["ridge_aug"],
        "hgb_aug_pred": preds["hgb_aug"],
        "ridge_base_pred": base_ridge,
        "hgb_base_pred": base_hgb,
    })
    pred_df.to_csv(DATA_OUT / "predictions.tsv", sep="\t", index=False, float_format="%.17g")

    # ---- serialize transforms + models ----
    joblib.dump({"ridge_aug": models["ridge_aug"], "hgb_aug": models["hgb_aug"]},
                attempt_dir / "models.joblib")
    joblib.dump(scaler, attempt_dir / "X_scaler.joblib")
    transforms = {
        "X_scaler": {"mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist()},
        "target_raw": {"train_mean": tr_mean_raw, "train_std": tr_std_raw},
    }
    atomic_write_json(attempt_dir / "transforms.json", transforms)

    metrics = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "n_eligible": n_eligible,
        "n_train": n_tr, "n_val": n_va, "n_test": n_te,
        "target_standardization": {"raw_train_mean": tr_mean_raw, "raw_train_std": tr_std_raw},
        "augmented_arms": aug_metrics,
        "baseline_arms": base_metrics,
        "val_descriptive": val_descriptive,
        "constant_baselines_raw": const_baselines,
        "raw_unit_deltas_aug_minus_base": raw_deltas_out,
        "primary_feature_gap": primary,
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(REPORT_OUT / "metrics.json", metrics)

    attempt_meta = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "attempt": args.attempt,
        "extract_attempt": args.extract_attempt,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_hashes": code_hashes,
        "extract_dir": str(xdir.resolve()),
        "models_sha256": sha256_file(attempt_dir / "models.joblib"),
        "scaler_sha256": sha256_file(attempt_dir / "X_scaler.joblib"),
        "thread_caps": thread_caps_record(),
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(attempt_dir / "attempt_meta.json", attempt_meta)

    append_ledger({
        "ts_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": "fit",
        "attempt": args.attempt,
        "attempt_dir": str(attempt_dir.resolve()),
        "status": "complete",
        "n_fits": 2,
        "fits": ["ridge_aug", "hgb_aug"],
        "models_sha256": sha256_file(attempt_dir / "models.joblib"),
        "elapsed_s": round(time.time() - t0, 2),
    })

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
