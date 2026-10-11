#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerical tests: affine rescale of Score_S -> MAE/MSE/R2/Spearman consequences.

Clearly labelled analytic / toy probe (no model prediction artifact available and
aligned for per-gene S), plus verification on the real 18947 defined genes using a
deterministic synthetic prediction (NOT a trained model).

Properties under affine transform T(x) = (x - a)/b,  b > 0:
  - T is strictly increasing -> Spearman invariant (order preserved) for both
    target-only and paired transform.
  - paired transform (target and prediction both T): MAE' = MAE / b, MSE' = MSE / b^2,
    R2 invariant.
  - target-only transform (prediction fixed): MAE/MSE/R2 are NOT invariant.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
REPORT_OUT = ROOT / "reports" / "breakthrough_20261002" / "batch08"
CANON = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"

def rankdata_average(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i, n = 0, len(x)
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks

def spearman(x, y):
    rx = rankdata_average(x); ry = rankdata_average(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")

def mae(p, y):
    return float(np.abs(p - y).mean())

def mse(p, y):
    return float(((p - y) ** 2).mean())

def r2(p, y):
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")

def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def main():
    g = pd.read_csv(CANON, sep="\t", low_memory=False)
    defined = g["Score_S"].notna()
    N = len(g); M = int(defined.sum())
    a = (N - M) / (N - 1)
    b = (M - 1) / (N - 1)

    rng = np.random.default_rng(20261003)
    n = 5000
    y = rng.uniform(a, 1.0, n)          # synthetic target in legacy Score_S domain
    p = np.clip(y + rng.normal(0.0, 0.05, n), 0.0, 1.0)   # synthetic prediction (labelled toy)
    T = lambda x: (x - a) / b
    yp = T(y); pp = T(p)

    checks = []
    def check(name, cond, detail=""):
        checks.append({"test": name, "pass": bool(cond), "detail": detail})

    # paired transform scaling
    check("MAE_paired_1over_b", close(mae(pp, yp), mae(p, y) / b, 1e-12),
          "MAE'=%.10f  MAE/b=%.10f" % (mae(pp, yp), mae(p, y) / b))
    check("MSE_paired_1over_b2", close(mse(pp, yp), mse(p, y) / (b ** 2), 1e-12),
          "MSE'=%.10f  MSE/b^2=%.10f" % (mse(pp, yp), mse(p, y) / (b ** 2)))
    check("R2_paired_invariant", close(r2(pp, yp), r2(p, y), 1e-9),
          "R2'=%.10f  R2=%.10f" % (r2(pp, yp), r2(p, y)))
    check("Spearman_paired_invariant", close(spearman(pp, yp), spearman(p, y), 1e-9),
          "sp'=%.10f  sp=%.10f" % (spearman(pp, yp), spearman(p, y)))
    check("Spearman_target_only_invariant", close(spearman(p, yp), spearman(p, y), 1e-9),
          "sp(target-only)=%.10f  sp=%.10f" % (spearman(p, yp), spearman(p, y)))
    # target-only is NOT invariant for MAE/MSE/R2
    check("MAE_target_only_NOT_invariant", not close(mae(p, yp), mae(p, y) / b, 1e-6),
          "target-only MAE=%.10f vs MAE/b=%.10f" % (mae(p, yp), mae(p, y) / b))
    check("MSE_target_only_NOT_invariant", not close(mse(p, yp), mse(p, y) / (b ** 2), 1e-6),
          "target-only MSE=%.10f vs MSE/b^2=%.10f" % (mse(p, yp), mse(p, y) / (b ** 2)))
    check("R2_target_only_NOT_invariant", not close(r2(p, yp), r2(p, y), 1e-6),
          "target-only R2=%.10f vs R2=%.10f" % (r2(p, yp), r2(p, y)))

    # ---- verify on the real 18947 defined genes ----
    s_legacy = g.loc[defined, "Score_S"].astype(float).to_numpy()
    s_affine = (s_legacy - a) / b
    # deterministic synthetic prediction (labelled, not a model): e.g. E-anchored surrogate
    s_pred = np.clip(g.loc[defined, "Score_E"].astype(float).to_numpy() * 0.5 + 0.25, 0, 1)
    sp_real = spearman(s_pred, s_legacy)
    sp_real_aff = spearman(T(s_pred), s_affine)
    checks.append({"test": "real_defined_Spearman_invariant_to_affine",
                   "pass": close(sp_real, sp_real_aff, 1e-9),
                   "detail": "sp_legacy=%.10f sp_affine(paired)=%.10f" % (sp_real, sp_real_aff)})

    # ordering identity among the three S conventions (no true ties in S_raw)
    # use full-precision recomputed S_raw from the audit table (S_raw_full column)
    audit = pd.read_csv(ROOT / "data" / "breakthrough_20261002" / "batch08" / "label_audit_all_genes.tsv",
                        sep="\t", low_memory=False)
    aud = audit[audit["S_defined_main"] == 1]
    s_raw_full = aud["S_raw_full"].astype(float).to_numpy()
    s_raw_6dec = aud["S_raw_6dec"].astype(float).to_numpy()
    ord_legacy = np.argsort(np.argsort(aud["Score_S"].astype(float).to_numpy(), kind="mergesort"), kind="mergesort")
    ord_affine = np.argsort(np.argsort(aud["S_affine"].astype(float).to_numpy(), kind="mergesort"), kind="mergesort")
    ord_raw_full = np.argsort(np.argsort(s_raw_full, kind="mergesort"), kind="mergesort")
    checks.append({"test": "legacy_affine_same_ordering",
                   "pass": bool(np.array_equal(ord_legacy, ord_affine)),
                   "detail": "affine is strictly monotonic in legacy"})
    checks.append({"test": "legacy_fullprecision_raw_same_ordering",
                   "pass": bool(np.array_equal(ord_legacy, ord_raw_full)),
                   "detail": "no full-precision S_raw ties => legacy ordinal rank == full-precision raw order"})
    # 6-decimal stored S_raw does NOT reproduce the exact legacy ordering (rounding ties)
    # how many genes are involved in 6-dec tie groups (rounding artifact)
    uniq, counts = np.unique(s_raw_6dec, return_counts=True)
    n_in_6dec_ties = int((counts[counts > 1]).sum())
    # a naive average-rank on 6-dec S_raw produces a different ordering vs legacy
    ord_6dec_avg = np.argsort(np.argsort(rankdata_average(s_raw_6dec), kind="mergesort"), kind="mergesort")
    n_order_changed = int((ord_legacy != ord_6dec_avg).sum())
    checks.append({"test": "sixdec_raw_does_NOT_reproduce_legacy_ordering",
                   "pass": n_order_changed > 0,
                   "detail": "genes in 6-dec tie groups=%d; order changes vs legacy under average-rank=%d"
                             % (n_in_6dec_ties, n_order_changed)})

    n_pass = sum(c["pass"] for c in checks)
    result = {"candidate_version": "v6.9.0-labelaudit.rc1",
              "a": a, "b": b,
              "note": "synthetic toy prediction clearly labelled; not a trained model",
              "checks": checks, "n_pass": n_pass, "n_total": len(checks)}
    (REPORT_OUT / "scale_invariance_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for c in checks:
        print(("PASS " if c["pass"] else "FAIL ") + c["test"] + " :: " + c["detail"])
    print("TOTAL %d/%d passed" % (n_pass, len(checks)))
    if n_pass != len(checks):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
