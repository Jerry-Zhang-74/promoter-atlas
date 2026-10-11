#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch10 saved-model replay + batch09 baseline replay (no refit).

1) Replays batch10 ridge_aug / hgb_aug from serialized models/scaler/transforms
   and verifies they match predictions.tsv (full eligible set).
2) Replays batch09 ridge_raw / hgb_raw from the batch09 serialized models/scaler/
   transforms on a deterministic small subset (5 train + 5 val + 5 test genes)
   and verifies they match batch09 predictions.tsv on that subset, proving the
   matched baselines are loaded/reproducible without refitting anything.
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

from b10_common import (ROOT, EXPR, DATA_OUT, MODEL_OUT, B9_PRED, B9_MODELS,
                        B9_SCALER, B9_TRANSFORMS, CANDIDATE_VERSION, BATCH,
                        reconstruct_s_raw, load_gene_universe, load_features45,
                        atomic_write_json)  # noqa: E402


def load_aug_x(extract_attempt=1):
    z = np.load(DATA_OUT / ("attempt_%03d_extract" % extract_attempt) / "features1024.npz")
    X1024 = z["X1024"].astype(np.float64)
    X45 = load_features45()
    return np.concatenate([X45, X1024], axis=1)


def main():
    fit_dir = MODEL_OUT / "attempt_001_fit"
    models = joblib.load(fit_dir / "models.joblib")
    scaler = joblib.load(fit_dir / "X_scaler.joblib")
    tf = json.load(open(fit_dir / "transforms.json", encoding="utf-8"))

    gene_gid, gene_split, comp_dense, n_prom, starts, _ = load_gene_universe()
    full_s = reconstruct_s_raw(EXPR)
    eligible = np.array([g in full_s for g in gene_gid])
    Xs = scaler.transform(load_aug_x()[eligible])
    s_raw = np.array([full_s[g] for g in gene_gid[eligible]])

    mu = tf["target_raw"]["train_mean"]
    sd = tf["target_raw"]["train_std"]
    p_ridge = models["ridge_aug"].predict(Xs) * sd + mu
    p_hgb = models["hgb_aug"].predict(Xs) * sd + mu

    pred = pd.read_csv(DATA_OUT / "predictions.tsv", sep="\t", low_memory=False)
    assert np.array_equal(pred["gene_id"].values, gene_gid[eligible])
    ok_ridge = np.allclose(p_ridge, pred["ridge_aug_pred"].values, atol=1e-8)
    ok_hgb = np.allclose(p_hgb, pred["hgb_aug_pred"].values, atol=1e-8)

    # ---- batch09 baseline replay on deterministic small subset ----
    b9_models = joblib.load(B9_MODELS)
    b9_scaler = joblib.load(B9_SCALER)
    b9_tf = json.load(open(B9_TRANSFORMS, encoding="utf-8"))
    X45 = load_features45()
    X45_e = b9_scaler.transform(X45[eligible])
    b9_mu = b9_tf["target_raw"]["train_mean"]
    b9_sd = b9_tf["target_raw"]["train_std"]
    b9_ridge = b9_models["ridge_raw"].predict(X45_e) * b9_sd + b9_mu
    b9_hgb = b9_models["hgb_raw"].predict(X45_e) * b9_sd + b9_mu

    b9_pred = pd.read_csv(B9_PRED, sep="\t", low_memory=False)
    b9_map = {row.gene_id: row for row in b9_pred.itertuples(index=False)}
    b9_ridge_saved = np.array([b9_map[g].ridge_raw_pred for g in gene_gid[eligible]])
    b9_hgb_saved = np.array([b9_map[g].hgb_raw_pred for g in gene_gid[eligible]])

    # deterministic subset: first 5 genes of each split (universe-eligible order)
    split_e = gene_split[eligible]
    sel = np.concatenate([np.flatnonzero(split_e == sp)[:5] for sp in ("train", "val", "test")])
    sel = np.sort(sel)
    ok_b9_ridge = np.allclose(b9_ridge[sel], b9_ridge_saved[sel], atol=1e-8)
    ok_b9_hgb = np.allclose(b9_hgb[sel], b9_hgb_saved[sel], atol=1e-8)

    example = pd.DataFrame({
        "gene_id": gene_gid[eligible][sel],
        "split": split_e[sel],
        "S_raw_full": s_raw[sel],
        "ridge_aug_pred": p_ridge[sel],
        "hgb_aug_pred": p_hgb[sel],
        "ridge_base_pred": b9_ridge_saved[sel],
        "hgb_base_pred": b9_hgb_saved[sel],
    })
    cols = ["gene_id", "split", "S_raw_full", "ridge_aug_pred", "hgb_aug_pred",
            "ridge_base_pred", "hgb_base_pred"]
    print(example[cols].to_string(index=False, float_format=lambda v: "%.6f" % v))

    out = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "replayed_eligible_rows": int(len(gene_gid[eligible])),
        "batch10_ridge_aug_replay_match": bool(ok_ridge),
        "batch10_hgb_aug_replay_match": bool(ok_hgb),
        "batch09_baseline_subset_replay": {
            "subset": "first 5 train + 5 val + 5 test genes (universe-eligible order)",
            "n_subset": int(len(sel)),
            "ridge_raw_match": bool(ok_b9_ridge),
            "hgb_raw_match": bool(ok_b9_hgb),
        },
    }
    atomic_write_json(DATA_OUT / "replay.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
