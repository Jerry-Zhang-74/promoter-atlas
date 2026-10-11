#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained invariant tests for the batch08 S label audit (no GPU, no training).

Run:  .venv/Scripts/python.exe -X utf8 runs/breakthrough_20261002/code/batch08/test_label_audit.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CANON = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"
AUDIT = ROOT / "data" / "breakthrough_20261002" / "batch08" / "label_audit_all_genes.tsv"
META = ROOT / "models" / "breakthrough_20261002" / "batch08" / "transform_metadata.json"

def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def main():
    g = pd.read_csv(CANON, sep="\t", low_memory=False)
    N = len(g); defined = g["Score_S"].notna(); M = int(defined.sum())
    a = (N - M) / (N - 1); b = (M - 1) / (N - 1)
    s = g.loc[defined, "Score_S"].astype(float)

    failures = []
    def check(name, cond):
        if not cond:
            failures.append(name)
        print(("PASS " if cond else "FAIL ") + name)

    check("N=58988", N == 58988)
    check("M=18947", M == 18947)
    check("S_defined_interval_low_6788", approx(s.min(), 0.678811, 1e-6))
    check("S_defined_interval_high_1", approx(s.max(), 1.0, 1e-9))
    check("a_formula", approx(a, (N - M) / (N - 1), 1e-15))
    check("b_formula", approx(b, (M - 1) / (N - 1), 1e-15))

    aud = pd.read_csv(AUDIT, sep="\t", low_memory=False)
    d = aud[aud["S_defined_main"] == 1]
    s_aff = d["S_affine"].astype(float)
    check("affine_range_low", approx(s_aff.min(), (0.678811 - a) / b, 1e-9))
    check("affine_range_high", approx(s_aff.max(), 1.0, 1e-9))
    check("affine_monotonic_preserves_order",
          bool(np.array_equal(np.argsort(s.to_numpy(), kind="mergesort"),
                              np.argsort(s_aff.to_numpy(), kind="mergesort"))))
    # undefined remain missing
    check("affine_missing_preserved", int(d["S_affine"].notna().sum()) == M and
          int(aud[aud["S_defined_main"] == 0]["S_affine"].notna().sum()) == 0)
    # defined rank quantile in [0,1]
    r = d["S_defined_rank"].astype(float)
    check("defined_rank_range_0_1", approx(r.min(), 0.0, 1e-9) and approx(r.max(), 1.0, 1e-9))

    meta = json.loads(META.read_text(encoding="utf-8"))
    check("metadata_a_b_match", approx(meta["affine_transform"]["a"], a, 1e-15)
          and approx(meta["affine_transform"]["b"], b, 1e-15))
    check("full_precision_zero_true_ties", meta["tie_audit_summary"]["full_precision_tie_groups"] == 0)

    if failures:
        print("FAILED:", failures)
        raise SystemExit(1)
    print("ALL PASS (%d checks)" % 14)

if __name__ == "__main__":
    main()
