#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch24 v6.18.0-hek-kernel.rc1 worker: fixed-kernel ridge, 5 folds x 2 recipes.

ZERO neural fits. <=10 kernel-ridge fits total (5 folds x 2 recipes).
The full label-free Gram is computed ONCE (all feature mappings are fixed and
per-sequence: no population-estimated parameters). Every attempt is
pre-registered append-only with actual PID / time / config SHA / input SHA and
its own immutable fold artifact directory.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b24_common as C
import batch24_kernels as KK

C.set_thread_limits(4)
PID = os.getpid()


def config_sha():
    spec = {
        "kernel_spec": KK.KERNEL_SPEC,
        "alpha": C.ALPHA,
        "recipes": list(C.RECIPES),
        "n_folds": C.N_FOLDS,
        "target": C.TARGET_COL,
        "input_sha": C.INPUT_SHA,
    }
    return C.sha256_text(json.dumps(spec, sort_keys=True, ensure_ascii=False))


def main():
    t0 = time.time()
    cfg_sha = config_sha()
    order, seqs, y, fold, grp, seen, k562 = C.load_rows()
    assert len(order) == 673
    assert len(np.unique(grp)) == 669
    assert len(set(map(len, seqs))) == 1 and len(seqs[0]) == 133

    # ---- full label-free Gram (computed ONCE; no target information) ---------
    res = KK.kernels_for(list(seqs))
    H1, H2 = res["H1"], res["H2"]
    np.save(C.DATA_OUT / "gram_H1_Sglobal.npy", H1)
    np.save(C.DATA_OUT / "gram_H2_combined.npy", H2)
    np.save(C.DATA_OUT / "gram_S_position.npy", res["S_position"])
    np.save(C.DATA_OUT / "gram_S_global.npy", res["S_global"])
    C.atomic_write_json(C.DATA_OUT / "kernel_manifest.json", {
        "generated_at_local": C.now_local(),
        "version": C.VERSION,
        "spec": KK.KERNEL_SPEC,
        "n_sequences": int(len(seqs)),
        "input_sha": C.INPUT_SHA,
        "label_free": True,
        "population_estimated_parameters": False,
        "note": ("the full 673x673 Gram is computed once because every feature "
                 "mapping is fixed and per-sequence; no target labels, no fitted "
                 "normalization, no data-dependent vocabulary"),
        "gram_files": {
            "gram_H1_Sglobal.npy": C.sha256_file(C.DATA_OUT / "gram_H1_Sglobal.npy"),
            "gram_H2_combined.npy": C.sha256_file(C.DATA_OUT / "gram_H2_combined.npy"),
            "gram_S_position.npy": C.sha256_file(C.DATA_OUT / "gram_S_position.npy"),
            "gram_S_global.npy": C.sha256_file(C.DATA_OUT / "gram_S_global.npy"),
        },
    })

    gram = {"H1": H1, "H2": H2}
    ledger = C.REPORT_OUT / "attempts_ledger.jsonl"
    if ledger.exists():
        raise SystemExit("REFUSE: attempts_ledger.jsonl already exists (append-only; "
                         "never re-run fits)")

    oof = {"H1": np.full(len(order), np.nan), "H2": np.full(len(order), np.nan)}
    attempt = 0
    fit_records = []
    for recipe in C.RECIPES:
        K = gram[recipe]
        for f in range(C.N_FOLDS):
            attempt += 1
            tr = np.where(fold != f)[0]
            ho = np.where(fold == f)[0]
            C.append_jsonl(ledger, {
                "event": "preregister", "attempt_id": attempt, "recipe": recipe,
                "fold": f, "alpha": C.ALPHA, "pid": PID,
                "started_at": C.now_local(), "config_sha": cfg_sha,
                "input_sha": C.INPUT_SHA,
                "n_train": int(len(tr)), "n_holdout": int(len(ho)),
                "solver": "numpy.linalg.solve float64 (K_train + alpha I) w = y_centered",
            })
            ts = time.time()
            mu = float(y[tr].mean())
            yc = y[tr] - mu
            A = K[np.ix_(tr, tr)].astype(np.float64).copy()
            A[np.diag_indices_from(A)] += C.ALPHA
            assert np.allclose(A, A.T, atol=1e-12), "K_train not symmetric"
            w = np.linalg.solve(A, yc)
            resid = float(np.linalg.norm(A @ w - yc))
            rel = resid / max(1.0, float(np.linalg.norm(yc)))
            ph = K[np.ix_(ho, tr)] @ w + mu
            assert np.isfinite(w).all() and np.isfinite(ph).all()
            oof[recipe][ho] = ph
            outdir = C.MODEL_OUT / ("attempt_fold%d_%s" % (f, recipe))
            outdir.mkdir(parents=True, exist_ok=True)
            np.savez(outdir / "pred.npz",
                     prom_id=np.array([order[i] for i in ho], dtype=object),
                     fold=np.array([f] * len(ho)), pred=ph)
            np.savez(outdir / "weights.npz",
                     train_id=np.array([order[i] for i in tr], dtype=object),
                     w=w, train_mean=mu, alpha=C.ALPHA)
            stats = {
                "attempt_id": attempt, "recipe": recipe, "fold": f, "pid": PID,
                "alpha": C.ALPHA, "n_train": int(len(tr)), "n_holdout": int(len(ho)),
                "train_mean": mu, "residual_norm": resid, "relative_residual": rel,
                "elapsed_s": time.time() - ts, "config_sha": cfg_sha,
                "input_sha": C.INPUT_SHA,
                "pred_sha256": C.sha256_file(outdir / "pred.npz"),
                "weights_sha256": C.sha256_file(outdir / "weights.npz"),
                "finite": bool(np.isfinite(w).all() and np.isfinite(ph).all()),
                "zero_neural_fits": True,
            }
            C.atomic_write_json(outdir / "stats.json", stats)
            C.append_jsonl(ledger, dict(stats, event="complete", ended_at=C.now_local(),
                                        artifact_dir=str(outdir)))
            fit_records.append(stats)

    # ---- OOF table ----------------------------------------------------------
    r1 = C.load_cached_r1(order)
    header = ["prom_id", "fold", "group_id", "source_seen", "hek_norm", "k562_norm",
              "pred_R1_cached", "pred_H1", "pred_H2"]
    with open(C.DATA_OUT / "oof_predictions_batch24.tsv", "w", encoding="utf-8",
              newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for i, p in enumerate(order):
            fh.write("\t".join([p, str(int(fold[i])), str(int(grp[i])),
                                str(int(seen[i])), repr(float(y[i])), repr(float(k562[i])),
                                repr(float(r1[i])), repr(float(oof["H1"][i])),
                                repr(float(oof["H2"][i]))]) + "\n")

    C.atomic_write_json(C.REPORT_OUT / "attempts_summary.json", {
        "generated_at_local": C.now_local(), "batch": C.BATCH, "version": C.VERSION,
        "n_attempts": attempt, "max_ridge_fits_budget": 10,
        "by_recipe": {r: C.N_FOLDS for r in C.RECIPES},
        "neural_full_fits": 0, "zero_neural_fits": True,
        "pid": PID, "config_sha": cfg_sha, "elapsed_s": time.time() - t0,
        "cached_R1_refits": 0,
    })
    print("fits done:", attempt, "elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
