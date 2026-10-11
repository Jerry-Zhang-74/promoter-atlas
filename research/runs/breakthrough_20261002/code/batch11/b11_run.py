#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch11 main run: four frozen scientific fits + metrics + four co-primary contrasts.

Four new scientific fits on 557-predictor matrices (45 baseline + one 512-feature
ablation block) against continuous raw S:
  A = old45 + positional3mer512 (tri bins)  -> Ridge + HGB
  B = old45 + global4mer512 (tetra)         -> Ridge + HGB
X StandardScaler fits eligible TRAIN only per representation; target standardized
on eligible TRAIN mean/std and predictions inverse-transformed to raw S units.

The batch10 full 1069-feature predictions (ridge_aug/hgb_aug) and the batch09
45-feature baselines (ridge_raw/hgb_raw) are LOADED/REPLAYED, never retrained.

Four co-primary paired component bootstrap contrasts (full - A, full - B) within
each of Ridge/HGB, test Spearman vs the common full-precision raw S truth,
n=2000, seed=20261011, percentile interval [0.625, 99.375].

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

from b11_common import (ROOT, EXPR, DATA_OUT, MODEL_OUT, REPORT_OUT, B9_PRED,
                        B10_PRED, CANDIDATE_VERSION, BATCH, SEED, N_BOOTSTRAP,
                        BOOT_LO_PCT, BOOT_HI_PCT, BOOT_TWO_SIDED_PCT, DELTA_MIN,
                        RIDGE, HGB, FEATURE_NAMES_45, FEATURE_NAMES_1024,
                        sha256_file, atomic_write_json, reconstruct_s_raw,
                        load_gene_universe, load_features45, load_features1024,
                        feature_arm_indices, spearman, r2_score, mae, mse,
                        thread_caps_record, N_EXPECTED_ELIGIBLE)  # noqa: E402

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
    for f in sorted(src_dir.glob("b11_*.py")):
        shutil.copy2(f, dst / f.name)
        hashes[f.name] = sha256_file(f)
    return hashes


def arm_matrix(X45, X1024, idx45, idx_1024):
    return np.concatenate([X45[:, idx45], X1024[:, idx_1024]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", type=int, default=1)
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
        X45 = load_features45()
        X1024 = load_features1024()
        idx45, tri, tetra = feature_arm_indices()
        X_A = arm_matrix(X45, X1024, idx45, tri)
        X_B = arm_matrix(X45, X1024, idx45, tetra)
        assert X_A.shape == (X45.shape[0], 557)
        assert X_B.shape == (X45.shape[0], 557)
        full_s = reconstruct_s_raw(EXPR)

        eligible_mask = np.array([gid in full_s for gid in gene_gid])
        n_eligible = int(eligible_mask.sum())
        assert n_eligible == N_EXPECTED_ELIGIBLE

        gene_id_e = gene_gid[eligible_mask]
        split_e = gene_split[eligible_mask]
        comp_e = comp_dense[eligible_mask]
        n_prom_e = n_prom[eligible_mask]
        X_A_e = X_A[eligible_mask]
        X_B_e = X_B[eligible_mask]
        s_raw = np.array([full_s[g] for g in gene_id_e], dtype=np.float64)
        assert np.isfinite(s_raw).all()

        tr = split_e == "train"
        va = split_e == "val"
        te = split_e == "test"
        n_tr, n_va, n_te = int(tr.sum()), int(va.sum()), int(te.sum())
        assert (n_tr, n_va, n_te) == (15004, 1898, 1867)

        # ---- X scalers: fit on eligible TRAIN only, per representation ----
        scaler_A = StandardScaler(); scaler_A.fit(X_A_e[tr]); Xs_A = scaler_A.transform(X_A_e)
        scaler_B = StandardScaler(); scaler_B.fit(X_B_e[tr]); Xs_B = scaler_B.transform(X_B_e)

        # ---- target standardization (train mean/std, population std) ----
        tr_mean_raw = float(s_raw[tr].mean())
        tr_std_raw = float(s_raw[tr].std(ddof=0))
        z_raw = (s_raw - tr_mean_raw) / tr_std_raw
        z_tr = z_raw[tr]

        # ---- four frozen scientific fits ----
        models = {}
        preds = {}

        m = Ridge(**RIDGE); m.fit(Xs_A[tr], z_tr)
        models["A_ridge"] = m
        preds["A_ridge"] = m.predict(Xs_A) * tr_std_raw + tr_mean_raw

        m = HistGradientBoostingRegressor(**HGB); m.fit(Xs_A[tr], z_tr)
        models["A_hgb"] = m
        preds["A_hgb"] = m.predict(Xs_A) * tr_std_raw + tr_mean_raw

        m = Ridge(**RIDGE); m.fit(Xs_B[tr], z_tr)
        models["B_ridge"] = m
        preds["B_ridge"] = m.predict(Xs_B) * tr_std_raw + tr_mean_raw

        m = HistGradientBoostingRegressor(**HGB); m.fit(Xs_B[tr], z_tr)
        models["B_hgb"] = m
        preds["B_hgb"] = m.predict(Xs_B) * tr_std_raw + tr_mean_raw

    # ---- load full (batch10) and baseline (batch09) predictions, no refit ----
    b10 = pd.read_csv(B10_PRED, sep="\t", low_memory=False)
    b10_map = {row.gene_id: row for row in b10.itertuples(index=False)}
    b9 = pd.read_csv(B9_PRED, sep="\t", low_memory=False)
    b9_map = {row.gene_id: row for row in b9.itertuples(index=False)}

    full_ridge = np.array([b10_map[g].ridge_aug_pred for g in gene_id_e], dtype=np.float64)
    full_hgb = np.array([b10_map[g].hgb_aug_pred for g in gene_id_e], dtype=np.float64)
    base_ridge = np.array([b9_map[g].ridge_raw_pred for g in gene_id_e], dtype=np.float64)
    base_hgb = np.array([b9_map[g].hgb_raw_pred for g in gene_id_e], dtype=np.float64)
    b9_s = np.array([b9_map[g].S_raw_full for g in gene_id_e], dtype=np.float64)
    assert np.allclose(b9_s, s_raw, atol=1e-12), "batch09 truth != batch11 reconstructed S"
    assert np.array_equal(np.array([b10_map[g].split for g in gene_id_e]), split_e)
    assert np.array_equal(np.array([b10_map[g].component_id for g in gene_id_e]), comp_e)

    # ---- point metrics on TEST (vs full raw truth) ----
    def arm_metrics(pred, y_true):
        return {
            "spearman_vs_raw_truth": spearman(pred[te], y_true[te]),
            "mae": mae(y_true[te], pred[te]),
            "mse": mse(y_true[te], pred[te]),
            "r2": r2_score(y_true[te], pred[te]),
        }

    all_preds = {
        "full_ridge": full_ridge, "full_hgb": full_hgb,
        "A_ridge": preds["A_ridge"], "A_hgb": preds["A_hgb"],
        "B_ridge": preds["B_ridge"], "B_hgb": preds["B_hgb"],
        "base_ridge": base_ridge, "base_hgb": base_hgb,
    }
    test_metrics = {k: arm_metrics(v, s_raw) for k, v in all_preds.items()}

    # ---- descriptive VAL Spearman ----
    val_spearman = {k: spearman(v[va], s_raw[va]) for k, v in all_preds.items()}
    # ---- descriptive TRAIN Spearman ----
    train_spearman = {k: spearman(v[tr], s_raw[tr]) for k, v in all_preds.items()}

    # ---- constant baselines (raw units, train mean/median on test) ----
    def baseline_metrics(y_tr, y_te):
        out = {}
        for name, const in [("train_mean", float(np.mean(y_tr))),
                            ("train_median", float(np.median(y_tr)))]:
            p = np.full(len(y_te), const)
            out[name] = {"mae": mae(y_te, p), "mse": mse(y_te, p), "r2": r2_score(y_te, p)}
        return out

    const_baselines = baseline_metrics(s_raw[tr], s_raw[te])

    # ---- paired component bootstrap (all arms on the SAME resample) ----
    comp_te = comp_e[te]
    uniq_comp = np.unique(comp_te)
    rng = np.random.default_rng(SEED)
    s_raw_te = s_raw[te]
    comp_list = {int(c): np.flatnonzero(comp_te == c) for c in uniq_comp}
    arm_test = {k: v[te] for k, v in all_preds.items()}

    # preallocate bootstrap Spearman per arm
    boot_rho = {k: np.empty(N_BOOTSTRAP) for k in all_preds}
    for b in range(N_BOOTSTRAP):
        sample = rng.choice(uniq_comp, size=len(uniq_comp), replace=True)
        idx = np.concatenate([comp_list[int(c)] for c in sample])
        truth_b = s_raw_te[idx]
        for k, at in arm_test.items():
            boot_rho[k][b] = spearman(at[idx], truth_b)

    def boot_summary(delta_arr, point):
        return {
            "point_estimate": float(point),
            "bootstrap_mean": float(delta_arr.mean()),
            "lo_pct": BOOT_LO_PCT,
            "hi_pct": BOOT_HI_PCT,
            "two_sided_pct": BOOT_TWO_SIDED_PCT,
            "lo": float(np.percentile(delta_arr, BOOT_LO_PCT)),
            "hi": float(np.percentile(delta_arr, BOOT_HI_PCT)),
            "n_resamples": N_BOOTSTRAP,
            "n_bootstrap_nan": int(np.isnan(delta_arr).sum()),
        }

    def contrast(full_key, arm_key, learner):
        d = boot_rho[full_key] - boot_rho[arm_key]
        point = spearman(arm_test[full_key], s_raw_te) - spearman(arm_test[arm_key], s_raw_te)
        lo = float(np.percentile(d, BOOT_LO_PCT))
        hi = float(np.percentile(d, BOOT_HI_PCT))
        return {
            "learner": learner,
            "full_arm": full_key,
            "ablated_arm": arm_key,
            "increment_of": ("B given A" if arm_key.startswith("A_") else "A given B"),
            "full_test_spearman": float(spearman(arm_test[full_key], s_raw_te)),
            "arm_test_spearman": float(spearman(arm_test[arm_key], s_raw_te)),
            "bootstrap": boot_summary(d, point),
            "delta": float(point),
            "adjusted_lo": lo,
            "adjusted_hi": hi,
            "meets_rule": bool(point >= DELTA_MIN and lo > 0),
        }

    primary = {
        "convention": ("Four co-primary comparisons; each uses a %.2f%% two-sided "
                       "percentile interval ([%.3f, %.3f] percentiles of the paired "
                       "component bootstrap delta distribution, Bonferroni-style 95%% "
                       "family across four)." % (BOOT_TWO_SIDED_PCT, BOOT_LO_PCT, BOOT_HI_PCT)),
        "acceptance_rule": "scoped conditional increment requires delta >= %.2f AND adjusted lower bound > 0" % DELTA_MIN,
        "contrasts": {
            "ridge_full_minus_A": contrast("full_ridge", "A_ridge", "Ridge"),
            "ridge_full_minus_B": contrast("full_ridge", "B_ridge", "Ridge"),
            "hgb_full_minus_A": contrast("full_hgb", "A_hgb", "HGB"),
            "hgb_full_minus_B": contrast("full_hgb", "B_hgb", "HGB"),
        },
    }

    # ---- secondary descriptive: A/B vs old45 baseline ----
    def secondary(full_key_base, arm_key, learner):
        d = boot_rho[arm_key] - boot_rho[full_key_base]
        point = spearman(arm_test[arm_key], s_raw_te) - spearman(arm_test[full_key_base], s_raw_te)
        return {
            "learner": learner,
            "arm": arm_key,
            "baseline": full_key_base,
            "arm_test_spearman": float(spearman(arm_test[arm_key], s_raw_te)),
            "baseline_test_spearman": float(spearman(arm_test[full_key_base], s_raw_te)),
            "bootstrap": boot_summary(d, point),
            "note": "descriptive only; NOT an extra confirmatory discovery",
        }

    secondary_descriptive = {
        "ridge_A_minus_base": secondary("base_ridge", "A_ridge", "Ridge"),
        "ridge_B_minus_base": secondary("base_ridge", "B_ridge", "Ridge"),
        "hgb_A_minus_base": secondary("base_hgb", "A_hgb", "HGB"),
        "hgb_B_minus_base": secondary("base_hgb", "B_hgb", "HGB"),
    }

    # ---- retention percentages (descriptive, unstable denominator guard) ----
    def retention(arm_key, base_key, full_key):
        num = spearman(arm_test[arm_key], s_raw_te) - spearman(arm_test[base_key], s_raw_te)
        den = spearman(arm_test[full_key], s_raw_te) - spearman(arm_test[base_key], s_raw_te)
        if abs(den) < 1e-3:
            return {"value": None, "guarded": True, "reason": "unstable denominator (|full-base gain|<1e-3)"}
        return {"value": float(num / den), "guarded": False,
                "numerator": float(num), "denominator": float(den)}

    retention = {
        "ridge_A": retention("A_ridge", "base_ridge", "full_ridge"),
        "ridge_B": retention("B_ridge", "base_ridge", "full_ridge"),
        "hgb_A": retention("A_hgb", "base_hgb", "full_hgb"),
        "hgb_B": retention("B_hgb", "base_hgb", "full_hgb"),
        "definition": "retention = (arm - old45 baseline) / (full - old45 baseline), test Spearman; descriptive only",
    }

    # ---- write per-gene predictions (universe-eligible order) ----
    pred_df = pd.DataFrame({
        "gene_id": gene_id_e,
        "split": split_e,
        "component_id": comp_e,
        "n_promoters": n_prom_e,
        "S_raw_full": s_raw,
        "A_ridge_pred": preds["A_ridge"],
        "A_hgb_pred": preds["A_hgb"],
        "B_ridge_pred": preds["B_ridge"],
        "B_hgb_pred": preds["B_hgb"],
        "full_ridge_pred": full_ridge,
        "full_hgb_pred": full_hgb,
        "base_ridge_pred": base_ridge,
        "base_hgb_pred": base_hgb,
    })
    pred_df.to_csv(DATA_OUT / "predictions.tsv", sep="\t", index=False, float_format="%.17g")

    # ---- serialize transforms + models ----
    joblib.dump({"A_ridge": models["A_ridge"], "A_hgb": models["A_hgb"],
                 "B_ridge": models["B_ridge"], "B_hgb": models["B_hgb"]},
                attempt_dir / "models.joblib")
    joblib.dump(scaler_A, attempt_dir / "X_scaler_A.joblib")
    joblib.dump(scaler_B, attempt_dir / "X_scaler_B.joblib")
    transforms = {
        "X_scaler_A": {"mean_": scaler_A.mean_.tolist(), "scale_": scaler_A.scale_.tolist()},
        "X_scaler_B": {"mean_": scaler_B.mean_.tolist(), "scale_": scaler_B.scale_.tolist()},
        "target_raw": {"train_mean": tr_mean_raw, "train_std": tr_std_raw},
        "n_features": {"A": int(X_A_e.shape[1]), "B": int(X_B_e.shape[1])},
    }
    atomic_write_json(attempt_dir / "transforms.json", transforms)

    metrics = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "n_eligible": n_eligible,
        "n_train": n_tr, "n_val": n_va, "n_test": n_te,
        "n_components_test": int(uniq_comp.size),
        "target_standardization": {"raw_train_mean": tr_mean_raw, "raw_train_std": tr_std_raw},
        "test_metrics": test_metrics,
        "train_spearman_descriptive": train_spearman,
        "val_spearman_descriptive": val_spearman,
        "constant_baselines_raw": const_baselines,
        "primary_conditional_increment": primary,
        "secondary_descriptive_vs_baseline": secondary_descriptive,
        "retention_descriptive": retention,
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(REPORT_OUT / "metrics.json", metrics)

    attempt_meta = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "attempt": args.attempt,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_hashes": code_hashes,
        "models_sha256": sha256_file(attempt_dir / "models.joblib"),
        "scaler_A_sha256": sha256_file(attempt_dir / "X_scaler_A.joblib"),
        "scaler_B_sha256": sha256_file(attempt_dir / "X_scaler_B.joblib"),
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
        "n_fits": 4,
        "fits": ["A_ridge", "A_hgb", "B_ridge", "B_hgb"],
        "models_sha256": sha256_file(attempt_dir / "models.joblib"),
        "elapsed_s": round(time.time() - t0, 2),
    })

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

