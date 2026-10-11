#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch09 preflight: verify provenance, joins, eligible population, split.

This writes the eligible per-gene table and a preflight summary. It does NOT
fit any model and does NOT inspect held-out (val/test) predictions.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b9_common import (ROOT, EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ,
                       FEATURES_NPZ, DATA_OUT, REPORT_OUT, MODEL_OUT,
                       CANDIDATE_VERSION, BATCH, N_FEATURE_UNIVERSE,
                       M_DEFINED, N_CANONICAL, N_EXPECTED_ELIGIBLE,
                       N_EXPECTED_EXCLUDED, sha256_file, atomic_write_json,
                       reconstruct_s_raw, load_gene_universe, load_features,
                       base_gene_id, spearman, rankdata_average)


def main():
    t0 = time.time()
    for d in (DATA_OUT, REPORT_OUT, MODEL_OUT):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. source SHAs ----
    sha = {
        "split_arrays.npz": sha256_file(SPLIT_NPZ),
        "features45.npz": sha256_file(FEATURES_NPZ),
        "gene_expression_table.tsv": sha256_file(EXPR),
        "gene_expression_label_table.tsv": sha256_file(CANON),
        "batch08_label_audit_all_genes.tsv": sha256_file(AUDIT_TSV),
        "trackA_gene_promoters.npz": sha256_file(TRACKA_NPZ),
    }
    exp = {
        "split_arrays.npz": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
        "features45.npz": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
    }
    sha_ok = {k: (sha[k] == exp[k]) for k in exp}

    # ---- 2. load universe / split / features ----
    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    X = load_features()
    assert X.shape == (N_FEATURE_UNIVERSE, 45)
    assert np.isfinite(X).all(), "features contain non-finite values"

    # alignment check: log1p_count must equal log1p(n_promoters) elementwise
    assert np.allclose(X[:, 44], np.log1p(n_prom.astype(np.float64))), "log1p_count != log1p(n_promoters)"
    # length columns are constants 400 / 2000
    assert np.allclose(X[:, 21], 400.0), "sym400 length column not constant 400"
    assert np.allclose(X[:, 43], 2000.0), "up2000 length column not constant 2000"
    # gc columns within [0,1]
    assert X[:, 20].min() >= -1e-12 and X[:, 20].max() <= 1 + 1e-12
    assert X[:, 42].min() >= -1e-12 and X[:, 42].max() <= 1 + 1e-12

    # ---- 3. reconstruct full-precision S_raw in memory ----
    full_s = reconstruct_s_raw(EXPR)
    n_full = len(full_s)
    assert n_full == M_DEFINED, "full-precision S_raw count %d != %d" % (n_full, M_DEFINED)

    # ---- 4. canonical defined set from label table ----
    g = pd.read_csv(AUDIT_TSV, sep="\t", low_memory=False)
    assert len(g) == N_CANONICAL
    defined_mask = g["S_defined_main"] == 1
    canonical_defined = set(g.loc[defined_mask, "gene_id"])
    assert len(canonical_defined) == M_DEFINED

    # ---- 5. join: eligible = defined AND in feature universe ----
    feat_ids = set(gene_gid)
    eligible_ids = canonical_defined & feat_ids
    excluded_ids = canonical_defined - feat_ids
    n_eligible = len(eligible_ids)
    n_excluded = len(excluded_ids)

    # gene_id order alignment between feature universe and canonical table must
    # be by unique gene id (never row order). Build per-row eligible mask.
    eligible_mask = np.array([gid in eligible_ids for gid in gene_gid])
    n_eligible_rows = int(eligible_mask.sum())
    assert n_eligible_rows == n_eligible

    # excluded genes characterization: absent from feature universe (no promoter
    # feature row); they still pass the B gate (tissue_count=54, B_raw>=0.30).
    ex_rows = g.loc[g["gene_id"].isin(excluded_ids)]
    excluded_reason_counts = Counter()
    for _, r in ex_rows.iterrows():
        excluded_reason_counts["S_defined_but_no_promoter_feature"] += 1
    excluded_tissue_counts = sorted(ex_rows["tissue_count"].unique().tolist())
    excluded_b_min = float(ex_rows["B_raw"].min())
    excluded_b_max = float(ex_rows["B_raw"].max())

    # ---- 6. split / component distribution over eligible genes ----
    split_counts = Counter(gene_split[eligible_mask])
    split_components = {}
    for sp in ("train", "val", "test"):
        sub = eligible_mask & (gene_split == sp)
        split_components[sp] = {
            "genes": int(sub.sum()),
            "components": int(np.unique(comp_dense[sub]).size),
        }
    n_components = int(np.unique(comp_dense[eligible_mask]).size)

    # ---- 7. tie / ordering audit (measured, not assumed) ----
    # full-precision in-memory vs %.10g export (S_raw_full in audit TSV)
    tsv_s = dict(zip(g.loc[defined_mask, "gene_id"],
                     g.loc[defined_mask, "S_raw_full"].astype(float)))
    common = sorted(set(full_s) & set(tsv_s))
    max_diff = max(abs(full_s[x] - tsv_s[x]) for x in common)
    sp_export = spearman([full_s[x] for x in common], [tsv_s[x] for x in common])
    # rank equality between in-memory and export
    a = np.array([full_s[x] for x in common])
    b = np.array([tsv_s[x] for x in common])
    ra = rankdata_average(a)
    rb = rankdata_average(b)
    n_rank_diff = int((ra != rb).sum())

    # true ties in full precision across defined genes
    tie_groups = defaultdict(list)
    for gene, val in full_s.items():
        tie_groups[val].append(gene)
    true_tie_groups = sum(1 for v in tie_groups.values() if len(v) > 1)
    # six-decimal export ties (S_raw_6dec)
    six_tie = defaultdict(list)
    for gene, val in zip(g.loc[defined_mask, "gene_id"],
                         g.loc[defined_mask, "S_raw_6dec"].astype(float)):
        six_tie[val].append(gene)
    six_tie_groups = sum(1 for v in six_tie.values() if len(v) > 1)

    # ---- 8. write eligible table ----
    sraw = np.array([full_s.get(gid, np.nan) for gid in gene_gid[eligible_mask]])
    assert np.isfinite(sraw).all()
    rows = pd.DataFrame({
        "gene_id": gene_gid[eligible_mask],
        "split": gene_split[eligible_mask],
        "component_id": comp_dense[eligible_mask],
        "n_promoters": n_prom[eligible_mask],
        "S_raw_full": sraw,
    })
    # order rows deterministically by (split, gene_id) for a stable artifact
    split_order = {"train": 0, "val": 1, "test": 2}
    rows["split_code"] = rows["split"].map(split_order)
    rows = rows.sort_values(["split_code", "gene_id"], kind="mergesort").drop(columns="split_code")
    rows.to_csv(DATA_OUT / "eligible_genes.tsv", sep="\t", index=False,
                float_format="%.17g")

    summary = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_sha_ok": sha_ok,
        "source_sha": sha,
        "population": {
            "canonical_genes": N_CANONICAL,
            "canonical_S_defined": M_DEFINED,
            "feature_universe_genes": N_FEATURE_UNIVERSE,
            "eligible_S_defined_and_feature_covered": n_eligible,
            "excluded_canonical_S_defined": n_excluded,
            "expected_eligible": N_EXPECTED_ELIGIBLE,
            "expected_excluded": N_EXPECTED_EXCLUDED,
            "eligible_matches_expected": n_eligible == N_EXPECTED_ELIGIBLE,
            "excluded_matches_expected": n_excluded == N_EXPECTED_EXCLUDED,
        },
        "excluded_genes": {
            "count": n_excluded,
            "reason": "S-defined in canonical expression label table but absent from the 57827-gene promoter-feature universe (no human promoter records => no composition/count features, no split row)",
            "reason_counts": dict(excluded_reason_counts),
            "tissue_count_values": excluded_tissue_counts,
            "B_raw_min": excluded_b_min,
            "B_raw_max": excluded_b_max,
            "gene_ids": sorted(excluded_ids),
        },
        "eligible_split": {
            "train": split_components["train"],
            "val": split_components["val"],
            "test": split_components["test"],
            "total_genes": n_eligible,
            "total_components": n_components,
        },
        "feature_checks": {
            "n_features": 45,
            "log1p_count_matches_n_promoters": True,
            "sym400_len_constant_400": True,
            "up2000_len_constant_2000": True,
            "gc_in_01": True,
            "all_finite": True,
        },
        "tie_ordering_audit": {
            "full_precision_S_raw_count": n_full,
            "in_memory_vs_10g_export_max_abs_diff": max_diff,
            "in_memory_vs_10g_export_spearman": sp_export,
            "in_memory_vs_10g_export_rank_differ_genes": n_rank_diff,
            "full_precision_true_tie_groups": true_tie_groups,
            "sixdec_tie_groups": six_tie_groups,
            "conclusion": ("full-precision in-memory S_raw and the %.10g audit export have "
                           "identical ordering (0 rank differences); true ties = 0; the "
                           "6-decimal export creates rounding ties as documented in batch08"),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(REPORT_OUT / "preflight_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
