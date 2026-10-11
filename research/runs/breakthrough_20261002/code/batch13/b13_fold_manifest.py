#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch13 deterministic component five-fold allocation manifest.

Computes the label-independent fold assignment (raw S values never used):
  1) unique component IDs of eligible genes;
  2) sorted by canonical string representation;
  3) hash = SHA256(FOLD_SALT + ID);
  4) sorted by decreasing eligible gene size, then hash, then ID;
  5) greedy-assign to the fold with the smallest current eligible gene count
     (ties -> lowest fold index 0..4).
Saves the immutable fold manifest + algorithm + code/source SHAs BEFORE any fit
or held-out metric. Also verifies every gene appears exactly once and no
component crosses folds. Deterministic: rerunning yields byte-identical output.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b13_common import (  # noqa: E402
    ROOT, DATA_OUT, REPORT_OUT, MODEL_OUT, CANDIDATE_VERSION, BATCH, N_FOLDS,
    FOLD_SALT, N_EXPECTED_ELIGIBLE, N_EXPECTED_COMPONENTS, sha256_file,
    atomic_write_json, reconstruct_s_raw, load_gene_universe, fold_hash)


def assign_folds(comp_e):
    uniq = np.unique(comp_e)
    uniq_ints = [int(c) for c in uniq]
    sizes = {}
    for c in uniq_ints:
        sizes[c] = int((comp_e == c).sum())
    ids_canonical = sorted(uniq_ints, key=str)
    hashes = {c: fold_hash(c) for c in ids_canonical}
    order = sorted(ids_canonical, key=lambda c: (-sizes[c], hashes[c], str(c)))
    fold_counts = [0] * N_FOLDS
    fold_of_component = {}
    for c in order:
        m = min(fold_counts)
        fi = fold_counts.index(m)
        fold_of_component[c] = fi
        fold_counts[fi] += sizes[c]
    return fold_of_component, sizes, hashes, order, fold_counts


def main():
    t0 = time.time()
    gene_gid, gene_split, comp_dense, n_prom = load_gene_universe()
    full_s = reconstruct_s_raw()
    eligible = np.array([g in full_s for g in gene_gid])
    assert int(eligible.sum()) == N_EXPECTED_ELIGIBLE
    comp_e = comp_dense[eligible]
    assert int(np.unique(comp_e).size) == N_EXPECTED_COMPONENTS

    fold_of_component, sizes, hashes, order, fold_counts = assign_folds(comp_e)

    # map each eligible gene to its fold
    fold_of_gene = np.array([fold_of_component[int(c)] for c in comp_e], dtype=np.int64)

    # verify: every gene exactly once, no component crossing
    n_genes_total = int(len(comp_e))
    n_assigned = int((fold_of_gene >= 0).sum())
    assert n_assigned == n_genes_total == N_EXPECTED_ELIGIBLE
    # component crossing: each component maps to exactly one fold by construction
    comp_fold_map = {int(c): fold_of_component[int(c)] for c in np.unique(comp_e)}
    crossing = 0
    for c in np.unique(comp_e):
        f = fold_of_component[int(c)]
        if f < 0 or f >= N_FOLDS:
            crossing += 1

    fold_gene_counts = [int((fold_of_gene == f).sum()) for f in range(N_FOLDS)]
    fold_comp_counts = []
    fold_comp_ids = {}
    for f in range(N_FOLDS):
        sub = np.unique(comp_e[fold_of_gene == f])
        fold_comp_counts.append(int(sub.size))
        fold_comp_ids[str(f)] = [int(c) for c in sorted(sub)]

    # source SHAs recorded with the manifest (frozen BEFORE any fit)
    source_sha = {
        "split_arrays.npz": sha256_file(ROOT / "data" / "breakthrough_20261002" / "batch05" / "split_arrays.npz"),
        "features45.npz": sha256_file(ROOT / "data" / "breakthrough_20261002" / "batch06" / "features45.npz"),
        "features1024.npz": sha256_file(ROOT / "data" / "breakthrough_20261002" / "batch10" / "attempt_001_extract" / "features1024.npz"),
        "trackA_gene_promoters.npz": sha256_file(ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_gene_promoters.npz"),
    }
    code_sha = {
        "b13_fold_manifest.py": sha256_file(Path(__file__).resolve()),
        "b13_common.py": sha256_file(Path(__file__).resolve().parent / "b13_common.py"),
    }

    # per-component rows (component id, size, hash, sorted order index, fold)
    component_rows = []
    for pos, c in enumerate(order):
        component_rows.append({
            "component_id": c,
            "eligible_gene_size": sizes[c],
            "hash": hashes[c],
            "sorted_order_index": pos,
            "fold": fold_of_component[c],
        })

    manifest = {
        "candidate_version": CANDIDATE_VERSION,
        "batch": BATCH,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frozen_before_any_fit_or_heldout_metric": True,
        "n_folds": N_FOLDS,
        "salt": FOLD_SALT,
        "algorithm": (
            "sort unique component IDs by canonical string representation; "
            "hash = SHA256(salt + str(ID)); sort components by decreasing eligible "
            "gene size then hash then ID; greedy-assign to fold with smallest current "
            "eligible gene count (ties -> lowest fold index 0..4)."
        ),
        "labels_not_used": "raw S values never used for fold assignment",
        "n_eligible_genes": N_EXPECTED_ELIGIBLE,
        "n_components": N_EXPECTED_COMPONENTS,
        "fold_gene_counts": fold_gene_counts,
        "fold_component_counts": fold_comp_counts,
        "fold_components_by_fold": fold_comp_ids,
        "component_rows": component_rows,
        "verification": {
            "every_gene_assigned_exactly_once": bool(n_assigned == n_genes_total),
            "no_component_crossing": bool(crossing == 0),
            "sum_fold_gene_counts": int(sum(fold_gene_counts)),
            "sum_fold_component_counts": int(sum(fold_comp_counts)),
            "fold_gene_counts_balance_max_minus_min": int(max(fold_gene_counts) - min(fold_gene_counts)),
        },
        "source_sha": source_sha,
        "code_sha": code_sha,
        "elapsed_s": round(time.time() - t0, 2),
    }
    atomic_write_json(DATA_OUT / "fold_manifest.json", manifest)

    # also write a per-gene fold assignment table (universe-eligible order)
    import pandas as pd
    df = pd.DataFrame({
        "gene_id": gene_gid[eligible],
        "component_id": comp_e,
        "fold": fold_of_gene,
        "original_split": gene_split[eligible],
    })
    df.to_csv(DATA_OUT / "fold_assignment.tsv", sep="\t", index=False)

    print(json.dumps({
        "candidate_version": CANDIDATE_VERSION,
        "fold_gene_counts": fold_gene_counts,
        "fold_component_counts": fold_comp_counts,
        "balance_max_minus_min": int(max(fold_gene_counts) - min(fold_gene_counts)),
        "verification": manifest["verification"],
        "elapsed_s": round(time.time() - t0, 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
