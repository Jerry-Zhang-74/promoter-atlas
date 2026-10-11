# -*- coding: utf-8 -*-
"""batch54 INDEPENDENT metric + group-bootstrap implementations.

Own implementation (no import of batch53 eval_replay/metrics) so first-model acceptance
never rests on the producer evaluator as sole proof. Definitions follow
runs/breakthrough_20261002/batch52/PROTOCOL.txt:

  activity: pooled Spearman rho; group-equal MAE; same-unit RMSE; top10 precision
            ceil(0.1*n) with STABLE-ID tie-breaking (deterministic).
  pair    : delta = short - long. group-equal MAE(delta); macro STRICT sign accuracy per
            eligible group (true-zero handled separately; a predicted TIE counts incorrect);
            pooled Spearman(delta_true, delta_pred) secondary.
  bootstrap: paired group-resamples RETAINING SAMPLING MULTIPLICITY (>=30 groups).

Nothing here fits, trains, differentiates or touches a GPU.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats


def _fin(a):
    a = np.asarray(a, dtype=np.float64)
    return np.isfinite(a)


def spearman_rho(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = _fin(x) & _fin(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan")
    rx = stats.rankdata(x, method="average")
    ry = stats.rankdata(y, method="average")
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if d <= 0:
        return float("nan")
    return float((rx * ry).sum() / d)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = _fin(y_true) & _fin(y_pred)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def group_equal_mae(y_true, y_pred, groups, return_counts=False):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    groups = np.asarray(groups)
    vals, n_used, n_dropped = [], 0, 0
    for g in dict.fromkeys(groups.tolist()):
        m = (groups == g) & _fin(y_true) & _fin(y_pred)
        if not m.any():
            n_dropped += 1
            continue
        vals.append(float(np.mean(np.abs(y_true[m] - y_pred[m]))))
        n_used += 1
    out = float(np.mean(vals)) if vals else float("nan")
    if return_counts:
        return out, {"n_groups_used": n_used, "n_groups_dropped_nonfinite": n_dropped}
    return out


def topk_precision_fixed_ids(y_true, y_pred, ids, frac=0.1):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ids = np.asarray(ids).astype(str)
    n = y_true.size
    if n == 0:
        return float("nan")
    k = int(math.ceil(frac * n))
    k = max(1, min(k, n))
    yp = np.where(_fin(y_pred), y_pred, -np.inf)
    yt = np.where(_fin(y_true), y_true, -np.inf)
    order_p = sorted(range(n), key=lambda i: (-yp[i], ids[i]))
    order_t = sorted(range(n), key=lambda i: (-yt[i], ids[i]))
    top_true = set(order_t[:k])
    hit = sum(1 for i in order_p[:k] if i in top_true)
    return float(hit / k)


def pair_group_values(delta_true, delta_pred, pair_groups):
    dt = np.asarray(delta_true, float); dp = np.asarray(delta_pred, float)
    pg = np.asarray(pair_groups)
    mae_by_group, sign_by_group = {}, {}
    n_true_zero, n_pred_tie = 0, 0
    for g in dict.fromkeys(pg.tolist()):
        m = (pg == g)
        t, p = dt[m], dp[m]
        n_true_zero += int(np.sum(t == 0))
        n_pred_tie += int(np.sum(p == 0))
        fin = _fin(t) & _fin(p)
        if fin.any():
            mae_by_group[g] = float(np.mean(np.abs(t[fin] - p[fin])))
        nz = fin & (t != 0)
        if nz.any():
            tt, pp = t[nz], p[nz]
            correct = int(np.sum((pp != 0) & (np.sign(pp) == np.sign(tt))))
            sign_by_group[g] = correct / int(nz.sum())
    aux = {"n_pairs": int(dt.size), "n_groups": len(mae_by_group),
           "n_true_zero_pairs": n_true_zero,
           "true_zero_rate": float(n_true_zero / dt.size) if dt.size else float("nan"),
           "n_pred_ties": n_pred_tie,
           "pred_tie_rate": float(n_pred_tie / dt.size) if dt.size else float("nan"),
           "n_sign_eligible_groups": len(sign_by_group)}
    return mae_by_group, sign_by_group, aux


def macro_sign_accuracy(sign_by_group):
    if not sign_by_group:
        return float("nan")
    return float(np.mean(list(sign_by_group.values())))


def group_mean_mae_from_values(mae_by_group):
    if not mae_by_group:
        return float("nan")
    return float(np.mean(list(mae_by_group.values())))


def group_bootstrap_paired(groups, maeA_by_group, maeB_by_group,
                           signA_by_group, signB_by_group,
                           n_boot=2000, seed=20261004, min_groups=30):
    groups = list(dict.fromkeys(groups))
    G = len(groups)
    if G < min_groups:
        return {"ok": False, "reason": "fewer than %d groups: %d" % (min_groups, G), "n_groups": G}
    maeA = np.array([maeA_by_group[g] for g in groups], float)
    maeB = np.array([maeB_by_group[g] for g in groups], float)
    sA = np.array([signA_by_group.get(g, np.nan) for g in groups], float)
    sB = np.array([signB_by_group.get(g, np.nan) for g in groups], float)
    rng = np.random.default_rng(seed)
    mae_a_minus_b, sign_b_minus_a = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, G, size=G)
        mae_a_minus_b.append(float(np.nanmean(maeA[pick]) - np.nanmean(maeB[pick])))
        sa, sb = sA[pick], sB[pick]
        m = np.isfinite(sa) & np.isfinite(sb)
        if m.any():
            sign_b_minus_a.append(float(np.mean(sb[m]) - np.mean(sa[m])))
        else:
            sign_b_minus_a.append(float("nan"))

    def ci(a):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        if a.size == 0:
            return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
        return {"mean": float(a.mean()), "lo": float(np.percentile(a, 2.5)),
                "hi": float(np.percentile(a, 97.5)), "n": int(a.size)}
    mab = ci(mae_a_minus_b)
    # improvement of B over A = MAE_A - MAE_B (positive => B has smaller pair MAE)
    return {"ok": True, "n_groups": G, "n_boot": n_boot, "seed": seed,
            "multiplicity_retained": True,
            "mae_A_minus_B": mab,
            "sign_B_minus_A": ci(sign_b_minus_a),
            "improvement_mae_B_over_A": {"mean": mab["mean"], "lo": mab["lo"], "hi": mab["hi"]}}


def fixed_pair_baselines(delta_true, pair_groups):
    dt = np.asarray(delta_true, float)
    pg = np.asarray(pair_groups)
    n = dt.size
    out = {}
    for name, val in (("zero_delta", 0.0), ("always_positive", 1.0),
                      ("always_negative", -1.0), ("length_only_shorter_lower", -1.0)):
        dp = np.full(n, val, float)
        mae_by, sign_by, _ = pair_group_values(dt, dp, pg)
        out[name] = {"group_mean_mae": group_mean_mae_from_values(mae_by),
                     "macro_sign_accuracy": macro_sign_accuracy(sign_by)}
    return out


def activity_bundle(y_true, y_pred, groups, ids):
    mae, cnt = group_equal_mae(y_true, y_pred, groups, return_counts=True)
    return {"rho": spearman_rho(y_true, y_pred),
            "group_mae": mae,
            "rmse": rmse(y_true, y_pred),
            "top10_precision_fixed_ids": topk_precision_fixed_ids(y_true, y_pred, ids),
            "n_rows": int(np.asarray(y_true).size),
            "group_counts": cnt}


def pair_bundle(delta_true, delta_pred, pair_groups):
    mae_by, sign_by, aux = pair_group_values(delta_true, delta_pred, pair_groups)
    return {"group_mean_mae": group_mean_mae_from_values(mae_by),
            "macro_sign_accuracy": macro_sign_accuracy(sign_by),
            "rho_pooled": spearman_rho(delta_true, delta_pred),
            **aux}
