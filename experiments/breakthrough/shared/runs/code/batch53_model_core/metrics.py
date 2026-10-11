# -*- coding: utf-8 -*-
"""Fixed evaluation metrics (batch52 PROTOCOL), shared by checkpoint selection and eval.

Activity: pooled Spearman, group-equal MAE, same-unit RMSE, top10 precision ceil(.1n).
Pair primary: group-equal MAE(delta); macro strict sign accuracy per eligible group
(true zero handled separately, predicted ties counted incorrect); pooled Spearman secondary.
NO metric may be tuned on TEST.
"""
from __future__ import annotations
import math

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # average ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def group_equal_mae(y_true, y_pred, groups) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    groups = np.asarray(groups)
    errs = []
    for g in dict.fromkeys(groups.tolist()):
        m = groups == g
        errs.append(np.mean(np.abs(y_true[m] - y_pred[m])))
    return float(np.mean(errs)) if errs else float("nan")


def top10_precision(y_true, y_pred, stable_ids=None) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = int(math.ceil(0.1 * n))
    if stable_ids is not None:
        order_p = sorted(range(n), key=lambda i: (-y_pred[i], stable_ids[i]))
        order_t = sorted(range(n), key=lambda i: (-y_true[i], stable_ids[i]))
    else:
        order_p = list(np.argsort(-y_pred, kind="mergesort"))
        order_t = list(np.argsort(-y_true, kind="mergesort"))
    top_true = set(order_t[:k])
    return float(sum(1 for i in order_p[:k] if i in top_true) / k)


def group_mean_pair_mae(delta_true, delta_pred, groups) -> float:
    """Primary checkpoint-selection metric: mean over groups of per-group MAE(delta)."""
    return group_equal_mae(delta_true, delta_pred, groups)


def macro_sign_accuracy(delta_true, delta_pred, groups) -> float:
    """Per eligible group strict sign accuracy (excl. all-zero-true groups), macro mean.
    Strict: sign(pred) == sign(true); predicted ties (0) are incorrect unless true is 0
    (true-zero handled separately and excluded here)."""
    dt = np.asarray(delta_true, float); dp = np.asarray(delta_pred, float)
    groups = np.asarray(groups)
    accs = []
    for g in dict.fromkeys(groups.tolist()):
        m = groups == g
        t, p = dt[m], dp[m]
        nz = t != 0
        if not nz.any():
            continue
        correct = 0
        for ti, pi in zip(t[nz], p[nz]):
            if pi != 0 and np.sign(pi) == np.sign(ti):
                correct += 1
        accs.append(correct / int(nz.sum()))
    return float(np.mean(accs)) if accs else float("nan")


def true_zero_rate(delta_true) -> float:
    dt = np.asarray(delta_true, float)
    return float(np.mean(dt == 0)) if len(dt) else float("nan")
