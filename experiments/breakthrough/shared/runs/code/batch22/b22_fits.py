#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3: predeclared READOUT comparison R0 / R1 / R2 (small CPU ridge fits).

R0 reference : outer-train MEAN of the target (no learned representation; bookkeeping).
R1 baseline  : whole measured-sequence composition features (mono/di/tri frequency,
               GC fraction, log1p length) -> Ridge(alpha=100).
R2 candidate : SAME R1 features + the frozen batch21 RecipeA seed20261023 pre-scalar
               -head pooled representation (128-d) -> Ridge(alpha=100).

Fixed budget: 5 outer group folds x 2 ridge recipes = 10 ridge fits. R0 is a mean,
not a fit. No inner search, no seed shopping, no end-to-end neural fitting.
Feature scaling is fit on the OUTER TRAIN rows only (constants kept safe).

Every fit attempt is registered in an APPEND-ONLY ledger BEFORE execution and the
per-fold outputs are cached to immutable attempt directories. Existing finished
attempts are loaded, never retrained.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from b22_common import (DATA_OUT, MODEL_OUT, REPORT_OUT, ALPHA, now_local,
                        atomic_write_json, sha256_file, set_thread_limits)

LEDGER = REPORT_OUT / "attempts_ledger.jsonl"
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


def feature_composition(seqs):
    """mono(4) + di(16) + tri(64) frequencies + GC fraction + log1p(length)."""
    X = np.zeros((len(seqs), 4 + 16 + 64 + 2), dtype=np.float64)
    for i, s in enumerate(seqs):
        s = s.upper(); L = len(s)
        mono = np.zeros(4); di = np.zeros(16); tri = np.zeros(64); gc = 0
        for j, c in enumerate(s):
            b = BASE.get(c)
            if b is None:
                continue
            mono[b] += 1
            if c in "GC":
                gc += 1
            if j <= L - 2:
                b2 = BASE.get(s[j + 1])
                if b2 is not None:
                    di[b * 4 + b2] += 1
            if j <= L - 3:
                b2 = BASE.get(s[j + 1]); b3 = BASE.get(s[j + 2])
                if b2 is not None and b3 is not None:
                    tri[(b * 4 + b2) * 4 + b3] += 1
        if L:
            mono /= L
            gc /= L
        if L >= 2:
            di /= (L - 1)
        if L >= 3:
            tri /= (L - 2)
        X[i, :4] = mono; X[i, 4:20] = di; X[i, 20:84] = tri
        X[i, 84] = gc; X[i, 85] = np.log1p(L)
    return X


def load_inputs():
    rows = list(csv.DictReader(open(DATA_OUT / "sequence_table.tsv", encoding="utf-8"),
                               delimiter="\t"))
    seqs = [r["sequence"].upper() for r in rows]
    y = np.array([float(r["hek_norm"]) for r in rows], dtype=np.float64)
    diag = np.array([float(r["k562_norm"]) for r in rows], dtype=np.float64)
    pids = [r["prom_id"] for r in rows]
    fa = list(csv.DictReader(open(DATA_OUT / "fold_assignment.tsv", encoding="utf-8"),
                             delimiter="\t"))
    fold_of = np.array([int(r["fold"]) for r in fa], dtype=np.int64)
    gid = np.array([int(r["group_id"]) for r in fa], dtype=np.int64)
    seen = np.array([int(r["source_seen"]) for r in fa], dtype=np.int64)
    z = np.load(DATA_OUT / "frozen_repr_A_seed20261023.npz", allow_pickle=True)
    H = z["H"].astype(np.float64)
    assert list(z["prom_id"]) == pids, "frozen representation row order mismatch"
    return rows, seqs, y, diag, pids, fold_of, gid, seen, H


def registry_add(rec):
    rec = dict(rec)
    rec["pid"] = os.getpid()
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def fit_ridge(Xtr, ytr, Xte, alpha=ALPHA):
    """Standardize on train (constants safe) then closed-form Ridge."""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    A = Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1])
    b = Ztr.T @ ytr
    w = np.linalg.solve(A, b)
    intercept = float(ytr.mean() - (Ztr.mean(axis=0) @ w))
    return Zte @ w + intercept, {"alpha": alpha, "train_mean": mu.tolist(),
                                 "train_scale": sd.tolist(), "coef": w.tolist(),
                                 "intercept": intercept}


def run():
    set_thread_limits(4)
    t0 = time.time()
    rows, seqs, y, diag, pids, fold_of, gid, seen, H = load_inputs()
    n = len(rows)
    k = int(fold_of.max()) + 1
    X1 = feature_composition(seqs)
    X2 = np.concatenate([X1, H], axis=1)
    config_sha = hashlib.sha256(
        json.dumps({"alpha": ALPHA, "folds": k, "r1_dim": X1.shape[1],
                    "r2_dim": X2.shape[1], "feature_def": "mono+di+tri+GC+log1plen",
                    "frozen_seed": 20261023}, sort_keys=True).encode()).hexdigest()
    input_sha = sha256_file(DATA_OUT / "sequence_table.tsv")

    oof = {r: np.full(n, np.nan) for r in ("R0", "R1", "R2")}
    n_fits_run = 0
    n_cached = 0
    for f in range(k):
        te = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        # R0: outer-train mean (bookkeeping, NOT a fit)
        oof["R0"][te] = float(y[tr].mean())
        for name, X in (("R1", X1), ("R2", X2)):
            adir = MODEL_OUT / ("attempt_fold%d_%s" % (f, name))
            pred_path = adir / "pred.npz"
            attempt_id = "b22_%s_fold%d" % (name, f)
            if pred_path.exists():
                d = np.load(pred_path, allow_pickle=True)
                oof[name][te] = d["pred"]
                n_cached += 1
                continue
            registry_add({"attempt_id": attempt_id, "recipe": name, "fold": f,
                          "start": now_local(), "reason": "predeclared ridge fit (alpha=100)",
                          "config_sha": config_sha, "input_sha": input_sha,
                          "n_train": int(len(tr)), "n_test": int(len(te)),
                          "dim": int(X.shape[1])})
            ts = time.time()
            pred, stats = fit_ridge(X[tr], y[tr], X[te])
            oof[name][te] = pred
            adir.mkdir(parents=True, exist_ok=True)
            np.savez(pred_path, test_idx=te, pred=pred, true=y[te])
            atomic_write_json(adir / "stats.json", {
                "attempt_id": attempt_id, "recipe": name, "fold": f,
                "n_train": int(len(tr)), "n_test": int(len(te)), "dim": int(X.shape[1]),
                "alpha": ALPHA, "elapsed_s": round(time.time() - ts, 2),
                "config_sha": config_sha, "input_sha": input_sha})
            np.savez(adir / "train_stats.npz", train_mean=np.array(stats["train_mean"]),
                     train_scale=np.array(stats["train_scale"]),
                     coef=np.array(stats["coef"]), intercept=stats["intercept"])
            n_fits_run += 1
    for name in oof:
        if not np.all(np.isfinite(oof[name])):
            raise RuntimeError("OOF incomplete for %s" % name)
    np.savez(DATA_OUT / "oof_predictions.npz", R0=oof["R0"], R1=oof["R1"], R2=oof["R2"],
             y=y, diag=diag, fold_of=fold_of, group_id=gid, source_seen=seen)
    with open(DATA_OUT / "oof_predictions.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["prom_id", "fold", "group_id", "source_seen", "hek_norm",
                    "k562_norm", "pred_R0", "pred_R1", "pred_R2"])
        for i, p in enumerate(pids):
            w.writerow([p, int(fold_of[i]), int(gid[i]), int(seen[i]),
                        y[i], diag[i], oof["R0"][i], oof["R1"][i], oof["R2"][i]])
    summary = {"generated_at_local": now_local(), "elapsed_s": round(time.time() - t0, 2),
               "n": n, "n_folds": k, "alpha": ALPHA,
               "r1_dim": int(X1.shape[1]), "r2_dim": int(X2.shape[1]),
               "n_ridge_fits_this_run": int(n_fits_run), "n_ridge_fits_cached": int(n_cached),
               "max_ridge_fits_budget": 5 * 2, "config_sha": config_sha, "input_sha": input_sha,
               "zero_neural_fits": True}
    atomic_write_json(REPORT_OUT / "fits_summary.json", summary)
    print("[b22_fits] n=%d folds=%d r1dim=%d r2dim=%d elapsed=%.1fs"
          % (n, k, X1.shape[1], X2.shape[1], time.time() - t0))
    return summary


if __name__ == "__main__":
    run()
