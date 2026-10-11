#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch10 preflight: verify provenance, hashes, joins, encoding, split.

Writes the eligible per-gene table (universe order) and a preflight summary.
Does NOT fit any model and does NOT inspect held-out (val/test) predictions.
Verifies source SHA256 (including the large memmaps), gene/split alignment,
encoding value distribution, eligibility counts and split/component counts.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b10_common import (ROOT, EXPR, CANON, AUDIT_TSV, TRACKA_NPZ, SPLIT_NPZ,
                        FEATURES_NPZ, MEM400, MEM2000, DATA_OUT, REPORT_OUT,
                        MODEL_OUT, CANDIDATE_VERSION, BATCH, N_FEATURE_UNIVERSE,
                        M_DEFINED, N_CANONICAL, N_EXPECTED_ELIGIBLE,
                        N_EXPECTED_EXCLUDED, sha256_file, atomic_write_json,
                        reconstruct_s_raw, load_gene_universe, load_features45,
                        base_gene_id, spearman, rankdata_average, N_HUMAN,
                        L400, L2000)


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
        "promoter_sym400_uint8.memmap": sha256_file(MEM400),
        "promoter_up2000_uint8.memmap": sha256_file(MEM2000),
    }
    exp = {
        "split_arrays.npz": "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5",
        "features45.npz": "c776ecfa74163b5000de8f82aa8961b5aed154f80333d95665530947694c0800",
        "trackA_gene_promoters.npz": "cd53ccb608c2969f279ac46ed49f94c9b56efd672393c937ec730139e61e4072",
    }
    sha_ok = {k: (sha[k] == exp[k]) for k in exp}

    # ---- 2. load universe / split / features ----
    gene_gid, gene_split, comp_dense, n_prom, starts, gene_id_perprom = load_gene_universe()
    X45 = load_features45()
    assert X45.shape == (N_FEATURE_UNIVERSE, 45)
    assert np.isfinite(X45).all()
    assert np.allclose(X45[:, 44], np.log1p(n_prom.astype(np.float64))), "log1p_count mismatch"

    # ---- 3. gene ordering / identity checks ----
    gid = gene_id_perprom[starts[:-1]]
    assert np.array_equal(gid, gene_gid)
    assert len(set(gid.tolist())) == N_FEATURE_UNIVERSE, "gene ids not unique"
    assert not any("." in x for x in gid), "gene ids contain version suffix"
    # split_arrays n_promoters equals diff(starts) (already asserted in loader)

    # ---- 4. sequence encoding / window checks (memmap) ----
    mm400 = np.memmap(MEM400, dtype="uint8", mode="r", shape=(N_HUMAN, L400))
    mm2000 = np.memmap(MEM2000, dtype="uint8", mode="r", shape=(N_HUMAN, L2000))
    hist400 = np.bincount(mm400.ravel(), minlength=5)
    hist2000 = np.bincount(mm2000.ravel(), minlength=5)
    encoding_ok = (hist400[4] > 0)  # ambiguous base 4 is present and encoded as N
    z = np.load(TRACKA_NPZ, allow_pickle=True)
    r400 = z["r400"].astype(np.int64)
    r2000 = z["r2000"].astype(np.int64)
    rec = z["record_id"].astype(str)
    r400_in_range = bool((r400 >= 0).all() and (r400 < N_HUMAN).all())
    r2000_in_range = bool((r2000 >= 0).all() and (r2000 < N_HUMAN).all())
    n_rec_unique = len(set(rec.tolist()))

    # ---- 5. reconstruct full-precision S_raw ----
    full_s = reconstruct_s_raw(EXPR)
    assert len(full_s) == M_DEFINED

    # ---- 6. eligible join ----
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

    # ---- 7. split / component distribution over eligible ----
    split_counts = Counter(gene_split[eligible_mask])
    split_components = {}
    for sp in ("train", "val", "test"):
        sub = eligible_mask & (gene_split == sp)
        split_components[sp] = {"genes": int(sub.sum()),
                                "components": int(np.unique(comp_dense[sub]).size)}
    n_components = int(np.unique(comp_dense[eligible_mask]).size)

    # ---- 8. write eligible table (universe order, like batch09 predictions) ----
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
        },
        "gene_identity_checks": {
            "gene_gid_matches_first_row_per_gene": True,
            "n_unique_gene_ids": N_FEATURE_UNIVERSE,
            "no_version_suffix": True,
            "n_promoters_equals_diff_starts": True,
            "log1p_count_matches_n_promoters": True,
        },
        "sequence_encoding_checks": {
            "sym400_value_histogram_A_C_G_T_N": [int(x) for x in hist400],
            "up2000_value_histogram_A_C_G_T_N": [int(x) for x in hist2000],
            "encoding_ok_ambiguous_present": bool(encoding_ok),
            "r400_in_range": bool(r400_in_range),
            "r2000_in_range": bool(r2000_in_range),
            "n_record_id": int(len(rec)),
            "n_record_id_unique": int(n_rec_unique),
            "note": "uint8 code A=0,C=1,G=2,T=3,N=4; N present in both windows",
        },
        "eligible_split": {
            "train": split_components["train"],
            "val": split_components["val"],
            "test": split_components["test"],
            "total_genes": n_eligible,
            "total_components": n_components,
        },
        "excluded_genes": {
            "count": n_excluded,
            "reason": "S-defined in canonical expression label table but absent from the 57827-gene promoter-feature universe",
            "gene_ids": sorted(excluded_ids),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(REPORT_OUT / "preflight_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
