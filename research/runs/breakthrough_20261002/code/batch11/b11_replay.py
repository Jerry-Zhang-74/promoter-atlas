#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch11 saved-model replay + full/baseline exact old-prediction matching (no refit).

1) Replays batch11 A_ridge/A_hgb/B_ridge/B_hgb from serialized models/scalers/
   transforms and verifies they match batch11 predictions.tsv (full eligible set).
2) Verifies the LOADED full (batch10 ridge_aug/hgb_aug) and baseline (batch09
   ridge_raw/hgb_raw) predictions match their frozen source TSVs exactly,
   proving the references are loaded/reproducible without refitting anything.
"""
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
import joblib  # noqa: E402

from b11_common import (ROOT, EXPR, DATA_OUT, MODEL_OUT, B9_PRED, B10_PRED,
                        CANDIDATE_VERSION, BATCH, reconstruct_s_raw,
                        load_gene_universe, load_features45, load_features1024,
                        feature_arm_indices, atomic_write_json)  # noqa: E402


def arm_matrix(X45, X1024, idx45, idx_1024):
    return np.concatenate([X45[:, idx45], X1024[:, idx_1024]], axis=1)


def main():
    with threadpool_limits(limits=4):
        fit_dir = MODEL_OUT / "attempt_001_fit"
        models = joblib.load(fit_dir / "models.joblib")
        scaler_A = joblib.load(fit_dir / "X_scaler_A.joblib")
        scaler_B = joblib.load(fit_dir / "X_scaler_B.joblib")
        tf = json.load(open(fit_dir / "transforms.json", encoding="utf-8"))

        gene_gid, gene_split, comp_dense, n_prom, starts, _ = load_gene_universe()
        X45 = load_features45()
        X1024 = load_features1024()
        idx45, tri, tetra = feature_arm_indices()
        full_s = reconstruct_s_raw(EXPR)
        eligible = np.array([g in full_s for g in gene_gid])
        X_A = arm_matrix(X45, X1024, idx45, tri)[eligible]
        X_B = arm_matrix(X45, X1024, idx45, tetra)[eligible]
        Xs_A = scaler_A.transform(X_A)
        Xs_B = scaler_B.transform(X_B)

        mu = tf["target_raw"]["train_mean"]
        sd = tf["target_raw"]["train_std"]

        p = {}
        p["A_ridge"] = models["A_ridge"].predict(Xs_A) * sd + mu
        p["A_hgb"] = models["A_hgb"].predict(Xs_A) * sd + mu
        p["B_ridge"] = models["B_ridge"].predict(Xs_B) * sd + mu
        p["B_hgb"] = models["B_hgb"].predict(Xs_B) * sd + mu

        pred = pd.read_csv(DATA_OUT / "predictions.tsv", sep="\t", low_memory=False)
        assert np.array_equal(pred["gene_id"].values, gene_gid[eligible])
        replay_match = {
            "A_ridge": bool(np.allclose(p["A_ridge"], pred["A_ridge_pred"].values, atol=1e-8)),
            "A_hgb": bool(np.allclose(p["A_hgb"], pred["A_hgb_pred"].values, atol=1e-8)),
            "B_ridge": bool(np.allclose(p["B_ridge"], pred["B_ridge_pred"].values, atol=1e-8)),
            "B_hgb": bool(np.allclose(p["B_hgb"], pred["B_hgb_pred"].values, atol=1e-8)),
        }
        replay_max_abs = {
            "A_ridge": float(np.max(np.abs(p["A_ridge"] - pred["A_ridge_pred"].values))),
            "A_hgb": float(np.max(np.abs(p["A_hgb"] - pred["A_hgb_pred"].values))),
            "B_ridge": float(np.max(np.abs(p["B_ridge"] - pred["B_ridge_pred"].values))),
            "B_hgb": float(np.max(np.abs(p["B_hgb"] - pred["B_hgb_pred"].values))),
        }

        # ---- full (batch10) exact old-prediction matching ----
        b10 = pd.read_csv(B10_PRED, sep="\t", low_memory=False)
        b10_map = {row.gene_id: row for row in b10.itertuples(index=False)}
        full_ridge = np.array([b10_map[g].ridge_aug_pred for g in gene_gid[eligible]])
        full_hgb = np.array([b10_map[g].hgb_aug_pred for g in gene_gid[eligible]])
        full_ridge_match = np.allclose(full_ridge, pred["full_ridge_pred"].values, atol=1e-12)
        full_hgb_match = np.allclose(full_hgb, pred["full_hgb_pred"].values, atol=1e-12)

        # ---- baseline (batch09) exact old-prediction matching ----
        b9 = pd.read_csv(B9_PRED, sep="\t", low_memory=False)
        b9_map = {row.gene_id: row for row in b9.itertuples(index=False)}
        base_ridge = np.array([b9_map[g].ridge_raw_pred for g in gene_gid[eligible]])
        base_hgb = np.array([b9_map[g].hgb_raw_pred for g in gene_gid[eligible]])
        base_ridge_match = np.allclose(base_ridge, pred["base_ridge_pred"].values, atol=1e-12)
        base_hgb_match = np.allclose(base_hgb, pred["base_hgb_pred"].values, atol=1e-12)

    out = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "replayed_eligible_rows": int(len(gene_gid[eligible])),
        "batch11_replay_match": replay_match,
        "batch11_replay_max_abs_diff": replay_max_abs,
        "full_batch10_exact_old_prediction_match": {
            "ridge_aug": bool(full_ridge_match),
            "hgb_aug": bool(full_hgb_match),
        },
        "baseline_batch09_exact_old_prediction_match": {
            "ridge_raw": bool(base_ridge_match),
            "hgb_raw": bool(base_hgb_match),
        },
        "note": "full and baseline loaded from frozen source TSVs (never refit)",
    }
    atomic_write_json(DATA_OUT / "replay.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

