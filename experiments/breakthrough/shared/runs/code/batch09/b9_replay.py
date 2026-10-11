#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch09 saved-model replay example.

Loads the serialized StandardScaler, four frozen models, the train-reference
rank function and the target transforms, then reproduces the four predictions
for the eligible genes and prints a short deterministic example table. This is
the canonical replay path for the serialized artifacts; it does not refit
anything and does not touch val/test labels for any fitting step.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b9_common import (MODEL_OUT, DATA_OUT, load_gene_universe, load_features,
                       reconstruct_s_raw, EXPR, set_thread_limits)
import joblib  # noqa: E402

set_thread_limits(4)


def emp_rank(sorted_train, x):
    n = len(sorted_train)
    x = np.asarray(x, dtype=np.float64)
    lt = np.searchsorted(sorted_train, x, side="left")
    eq = np.searchsorted(sorted_train, x, side="right") - lt
    return (lt.astype(np.float64) + 0.5 * eq.astype(np.float64)) / float(n)


def main():
    models = joblib.load(MODEL_OUT / "models.joblib")
    scaler = joblib.load(MODEL_OUT / "X_scaler.joblib")
    ref = np.load(MODEL_OUT / "rank_reference.npz")["sorted_train"]
    tf = json.load(open(MODEL_OUT / "transforms.json", encoding="utf-8"))

    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    X_all = load_features()
    full_s = reconstruct_s_raw(EXPR)
    eligible = np.array([g in full_s for g in gene_gid])
    Xs_e = scaler.transform(X_all[eligible])
    s_raw = np.array([full_s[g] for g in gene_gid[eligible]])
    F = emp_rank(ref, s_raw)

    p = {
        "ridge_raw": models["ridge_raw"].predict(Xs_e) * tf["target_raw"]["train_std"] + tf["target_raw"]["train_mean"],
        "ridge_rank": models["ridge_rank"].predict(Xs_e) * tf["target_rank"]["train_std"] + tf["target_rank"]["train_mean"],
        "hgb_raw": models["hgb_raw"].predict(Xs_e) * tf["target_raw"]["train_std"] + tf["target_raw"]["train_mean"],
        "hgb_rank": models["hgb_rank"].predict(Xs_e) * tf["target_rank"]["train_std"] + tf["target_rank"]["train_mean"],
    }
    df = pd.DataFrame({
        "gene_id": gene_gid[eligible],
        "split": gene_split[eligible],
        "component_id": comp_dense[eligible],
        "S_raw_full": s_raw,
        "rank_target_F": F,
        **p,
    })
    # deterministic sample for the replay example: 5 train + 5 val + 5 test genes
    sample = df.groupby("split", sort=False).head(5)
    sample = sample.sort_values(["split", "gene_id"])
    cols = ["gene_id", "split", "S_raw_full", "rank_target_F",
            "ridge_raw", "ridge_rank", "hgb_raw", "hgb_rank"]
    print(sample[cols].to_string(index=False, float_format=lambda v: "%.6f" % v))
    print("replayed rows:", len(df))


if __name__ == "__main__":
    main()
