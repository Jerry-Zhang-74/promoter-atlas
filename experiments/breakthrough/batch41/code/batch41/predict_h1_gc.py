#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch41 frozen CPU predictions: H1 spectrum-kernel READOUT (inference only) and the
fixed GC-fraction baseline, on the frozen 270bp insert input.

H1: reuse the batch24 fixed spectrum kernel (imported read-only) and the SAVED ridge
weights (K_new @ w + train_mean). NO solve/fit/calibration. 5-fold equal mean.
GC: GC fraction of the 270bp insert (no fitted sign).
Predictions are cached resumably and keyed by exact sequence SHA.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b41_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "runs/breakthrough_20261002/code/batch24"))
import batch24_kernels as KK  # noqa: E402


def load_input():
    import csv
    with open(C.DATA / "frozen_input_table.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return rows


def load_train():
    """673 training sequences and fold assignment (batch22, read-only)."""
    import csv
    seqs, order = {}, []
    with open(C.SEQ_TABLE, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            order.append(r["prom_id"])
            seqs[r["prom_id"]] = r["sequence"].strip().upper()
    fold = {}
    with open(C.FOLD_TABLE, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            fold[r["prom_id"]] = int(r["fold"])
    return order, seqs, fold


def feature_matrix(seqs, k):
    F = np.zeros((len(seqs), 4 ** k), dtype=np.float64)
    for i, s in enumerate(seqs):
        F[i] = KK.feature_vector(s, k)
    return F


def main():
    t0 = time.time()
    C.set_thread_limits(4)
    rows = load_input()
    seqs_q = [r["sequence"] for r in rows]
    shas_q = [r["sequence_sha256"] for r in rows]
    order, seqs_t, fold = load_train()
    idx = {p: i for i, p in enumerate(order)}
    assert len(order) == 673

    T4 = feature_matrix([seqs_t[p] for p in order], 4)
    T5 = feature_matrix([seqs_t[p] for p in order], 5)

    # verify the saved H1 Gram equals the recomputed Gram (feature tracing check)
    G = 0.5 * (T4 @ T4.T) + 0.5 * (T5 @ T5.T)
    saved = np.load(C.GRAM_H1)
    gram_max_abs_diff = float(np.max(np.abs(G - saved)))

    Q4 = feature_matrix(seqs_q, 4)
    Q5 = feature_matrix(seqs_q, 5)

    h1 = np.zeros((C.N_FOLDS, len(rows)), dtype=np.float64)
    fold_meta = []
    for f in range(C.N_FOLDS):
        wp = C.B24_MODELS / ("attempt_fold%d_H1/weights.npz" % f)
        z = np.load(wp, allow_pickle=True)
        train_ids = [str(x) for x in z["train_id"]]
        w = z["w"].astype(np.float64)
        mu = float(z["train_mean"])
        ti = np.array([idx[p] for p in train_ids], dtype=np.int64)
        assert len(ti) == len(w)
        Kq = 0.5 * (Q4 @ T4[ti].T) + 0.5 * (Q5 @ T5[ti].T)
        h1[f] = Kq @ w + mu
        fold_meta.append({"fold": f, "n_train": int(len(ti)), "train_mean": mu,
                          "weights_sha256": C.sha256_file(wp)})
    h1_mean = h1.mean(axis=0)

    gc = np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs_q], dtype=np.float64)

    # cache (resumable, keyed by exact sequence SHA)
    code_sha = C.sha256_file(Path(__file__))
    cache = {"model": "H1_batch24_spectrum_kernel_ridge",
             "input_sha": C.sha256_file(C.DATA / "frozen_input_table.tsv"),
             "code_sha256": code_sha, "gram_max_abs_diff": gram_max_abs_diff,
             "fold_meta": fold_meta,
             "pred": {shas_q[i]: {"h1_folds": [float(h1[f][i]) for f in range(C.N_FOLDS)],
                                  "h1_mean": float(h1_mean[i]),
                                  "gc": float(gc[i]),
                                  "h1_fold_spread": float(h1[:, i].std(ddof=0))}
                      for i in range(len(rows))}}
    C.atomic_write_json(C.DATA / "cache/h1/preds.json", cache)
    print(json.dumps({"n": len(rows), "gram_max_abs_diff": gram_max_abs_diff,
                      "h1_mean_min": float(h1_mean.min()), "h1_mean_max": float(h1_mean.max()),
                      "gc_min": float(gc.min()), "gc_max": float(gc.max()),
                      "elapsed_s": round(time.time() - t0, 2)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
