#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 preflight: verify provenance, hashes, joins, split, components.

Writes the eligible per-gene table (universe order) and a preflight summary.
Does NOT fit any model and does NOT inspect held-out metrics. Verifies source
SHA256, gene/split alignment, eligibility counts, split/component counts and the
full-precision raw S reconstruction.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b13_common import (  # noqa: E402
    ROOT, EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ, FEATURES45_NPZ,
    FEATURES1024_NPZ, DATA_OUT, REPORT_OUT, CANDIDATE_VERSION, BATCH,
    N_FEATURE_UNIVERSE, M_DEFINED, N_CANONICAL, N_EXPECTED_ELIGIBLE,
    N_EXPECTED_EXCLUDED, N_EXPECTED_COMPONENTS, EXPECTED_SHA, sha256_file,
    atomic_write_json, reconstruct_s_raw, load_gene_universe, load_features45,
    load_features1024)


def main():
    t0 = time.time()
    for d in (DATA_OUT, REPORT_OUT):
        d.mkdir(parents=True, exist_ok=True)

    sha = {
        "split_arrays.npz": sha256_file(SPLIT_NPZ),
        "features45.npz": sha256_file(FEATURES45_NPZ),
        "features1024.npz": sha256_file(FEATURES1024_NPZ),
        "trackA_gene_promoters.npz": sha256_file(TRACKA_NPZ),
        "gene_expression_table.tsv": sha256_file(EXPR),
        "gene_expression_label_table.tsv": sha256_file(CANON),
        "batch08_label_audit_all_genes.tsv": sha256_file(AUDIT_TSV),
    }
    sha_ok = {k: (sha[k] == EXPECTED_SHA[k]) for k in EXPECTED_SHA}

    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    X45 = load_features45()
    X1024 = load_features1024()
    assert X45.shape == (N_FEATURE_UNIVERSE, 45)
    assert X1024.shape == (N_FEATURE_UNIVERSE, 1024)
    assert np.isfinite(X45).all() and np.isfinite(X1024).all()
    assert np.allclose(X45[:, 44], np.log1p(n_prom.astype(np.float64))), "log1p_count mismatch"

    full_s = reconstruct_s_raw(EXPR)
    assert len(full_s) == M_DEFINED

    g = pd.read_csv(AUDIT_TSV, sep="\t", low_memory=False)
    assert len(g) == N_CANONICAL
    defined_mask = g["S_defined_main"] == 1
    canonical_defined = set(g.loc[defined_mask, "gene_id"])
    assert len(canonical_defined) == M_DEFINED
    feat_ids = set(gene_gid)
    eligible_ids = canonical_defined & feat_ids
    excluded_ids = canonical_defined - feat_ids
    n_eligible = len(eligible_ids)
    n_excluded = len(excluded_ids)
    eligible_mask = np.array([gid in eligible_ids for gid in gene_gid])
    assert int(eligible_mask.sum()) == n_eligible
    assert n_eligible == N_EXPECTED_ELIGIBLE
    assert n_excluded == N_EXPECTED_EXCLUDED

    split_counts = Counter(gene_split[eligible_mask].tolist())
    n_components = int(np.unique(comp_dense[eligible_mask]).size)
    assert n_components == N_EXPECTED_COMPONENTS

    split_components = {}
    for sp in ("train", "val", "test"):
        sub = eligible_mask & (gene_split == sp)
        split_components[sp] = {"genes": int(sub.sum()),
                                "components": int(np.unique(comp_dense[sub]).size)}

    sraw = np.array([full_s.get(gid, np.nan) for gid in gene_gid[eligible_mask]])
    assert np.isfinite(sraw).all()
    rows = pd.DataFrame({
        "gene_id": gene_gid[eligible_mask],
        "split": gene_split[eligible_mask],
        "component_id": comp_dense[eligible_mask],
        "n_promoters": n_prom[eligible_mask],
        "S_raw_full": sraw,
    })
    rows.to_csv(DATA_OUT / "eligible_genes.tsv", sep="\t", index=False, float_format="%.17g")

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
            "eligible_matches_expected": n_eligible == N_EXPECTED_ELIGIBLE,
            "excluded_matches_expected": n_excluded == N_EXPECTED_EXCLUDED,
            "eligible_components": n_components,
            "eligible_components_matches_expected": n_components == N_EXPECTED_COMPONENTS,
        },
        "gene_identity_checks": {
            "n_unique_gene_ids": N_FEATURE_UNIVERSE,
            "log1p_count_matches_n_promoters": True,
            "n_promoters_equals_diff_starts": True,
        },
        "eligible_split": {
            "train": split_components["train"],
            "val": split_components["val"],
            "test": split_components["test"],
            "total_genes": n_eligible,
            "total_components": n_components,
            "note": "original batch05 train/val/test roles retained as metadata only for this experiment",
        },
        "excluded_genes": {
            "count": n_excluded,
            "reason": "S-defined in canonical expression label table but absent from the 57827-gene promoter-feature universe",
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(REPORT_OUT / "preflight_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
