# -*- coding: utf-8 -*-
"""batch30 metric helpers (pure numpy, no fits).
Provenance: logic from runs/breakthrough_20261002/code/batch26/b26_metrics.py
  sha256 d10f1ea460551524f20afac2f31635b1e018769e8de4d198b0e5a1942d1995b2 (READ-ONLY original).
"""
from __future__ import annotations
import numpy as np


def _rank(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def r2(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ss = ((a - b) ** 2).sum(); tot = ((a - a.mean()) ** 2).sum()
    return float(1.0 - ss / tot) if tot > 0 else float("nan")


def group_balanced_mae(y_true, y_pred, groups):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    vals = []
    for g in np.unique(groups):
        m = groups == g
        vals.append(np.mean(np.abs(y_true[m] - y_pred[m])))
    return float(np.mean(vals)), int(len(vals))


def top_decile(y_true, y_pred, keys, frac=0.1):
    n = len(y_true)
    k = int(np.ceil(frac * n))
    h = np.array([int(__import__("hashlib").sha256(str(x).encode()).hexdigest()[:16], 16) for x in keys])
    order = np.lexsort((h, -np.asarray(y_pred, dtype=np.float64)))
    pred_top = set(order[:k].tolist())
    order_true = np.lexsort((h, -np.asarray(y_true, dtype=np.float64)))
    true_top = set(order_true[:k].tolist())
    inter = len(pred_top & true_top)
    base = k / n
    return {"k": k, "n": n, "intersection": inter, "precision": inter / k,
            "baseline_precision": base, "enrichment": (inter / k) / base if base > 0 else float("nan")}


def group_bootstrap(pred_a, pred_b, y, groups, n=2000, seed=20261030):
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    deltas = []
    for _ in range(n):
        pick = rng.randint(0, len(ug), len(ug))
        rows = np.concatenate([idx_by_g[ug[i]] for i in pick])
        ra = spearman(pred_a[rows], y[rows])
        rb = spearman(pred_b[rows], y[rows])
        if np.isfinite(ra) and np.isfinite(rb):
            deltas.append(ra - rb)
    d = np.asarray(deltas)
    return {"n_resamples": int(len(d)),
            "delta_rho_mean": float(d.mean()) if len(d) else float("nan"),
            "delta_rho_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else None}
