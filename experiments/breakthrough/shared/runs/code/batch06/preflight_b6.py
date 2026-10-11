#!/usr/bin/env python3
"""Batch06 preflight: validate frozen split/labels/IDs/features and freeze config."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b6_common import (  # noqa: E402
    REPORT, MODEL_DIR, DATA, DATA05, RUNS, atomic_write_json, sha256_file,
    ALPHA_GRID, N_FOLDS, SEEDS, CANDIDATE,
)
from b6_data import load_b6, load_features, make_folds_components  # noqa: E402

SPLIT_ARRAYS = DATA05 / "split_arrays.npz"
EXPECTED_SPLIT_ARRAYS_SHA = "4fbfccf43038fbe2211c2a21f113b11b38d92cf410bc3591c6bb3e3a42d8a6f5"
LABEL_TABLE = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"


def main():
    (gene_id, r400, score_e, starts, gene_split, gene_score_e,
     n_genes, comp_dense, n_prom, gene_gid) = load_b6()

    checks = {}
    split_sha = sha256_file(SPLIT_ARRAYS)
    checks["split_arrays_sha_matches_expected"] = (split_sha == EXPECTED_SPLIT_ARRAYS_SHA)
    checks["split_arrays_sha"] = split_sha

    checks["n_genes"] = int(n_genes)
    checks["n_starts_minus_1"] = int(len(starts) - 1)
    checks["gene_gid_unique"] = bool(len(np.unique(gene_gid)) == n_genes)
    checks["starts_monotonic"] = bool(np.all(np.diff(starts) > 0))
    checks["starts_first_zero"] = bool(starts[0] == 0)
    checks["n_promoters_all_positive"] = bool(np.all(n_prom >= 1))
    checks["n_promoters_sum_matches"] = bool(n_prom.sum() == len(gene_id))

    checks["gene_split_len"] = bool(len(gene_split) == n_genes)
    checks["comp_dense_len"] = bool(len(comp_dense) == n_genes)
    checks["n_prom_len"] = bool(len(n_prom) == n_genes)
    checks["gene_gid_len"] = bool(len(gene_gid) == n_genes)
    checks["split_values"] = sorted(np.unique(gene_split).tolist())

    lab = pd.read_csv(LABEL_TABLE, sep="\t")
    lab = lab[["gene_id", "Score_E"]].dropna()
    lab_map = dict(zip(lab["gene_id"].astype(str), lab["Score_E"].astype(float)))
    matched = 0
    max_abs_diff = 0.0
    for gid, val in zip(gene_gid, gene_score_e):
        if gid in lab_map:
            matched += 1
            max_abs_diff = max(max_abs_diff, abs(float(lab_map[gid]) - float(val)))
    checks["label_table_matched_genes"] = matched
    checks["label_table_max_abs_diff"] = float(max_abs_diff)
    checks["label_table_diff_tiny"] = bool(max_abs_diff < 1e-4)

    sa = pd.read_csv(DATA05 / "split_assignment.tsv", sep="\t")
    sa = sa.sort_values("gene_id").reset_index(drop=True)
    order = np.argsort(gene_gid, kind="mergesort")
    comp_sorted_gid = comp_dense[order]
    checks["comp_dense_matches_assignment"] = bool(
        np.array_equal(comp_sorted_gid, sa["component_id"].to_numpy(dtype=np.int64)))
    checks["comp_dense_dense_ids"] = bool(np.array_equal(
        np.unique(comp_dense), np.arange(len(np.unique(comp_dense)), dtype=np.int64)))
    checks["n_components"] = int(len(np.unique(comp_dense)))
    checks["largest_component_genes"] = int(max(np.bincount(comp_dense)))

    train_idx = np.flatnonzero(gene_split == "train")
    fold_id, assign = make_folds_components(comp_dense, train_idx, n_folds=N_FOLDS, seed=20261011)
    comp2fold = {}
    span = False
    for c, f in zip(comp_dense[train_idx], fold_id):
        if c in comp2fold and comp2fold[c] != f:
            span = True
            break
        comp2fold[c] = f
    checks["fold_component_span"] = not span
    checks["n_train_genes"] = int(len(train_idx))
    checks["fold_sizes_train"] = {int(f): int((fold_id == f).sum()) for f in range(N_FOLDS)}
    checks["fold_components"] = {int(f): int(len(set(comp_dense[train_idx][fold_id == f])))
                                 for f in range(N_FOLDS)}

    comp44_all, log1p_count, gc_gene, n_prom_f = load_features()
    checks["features_comp44_shape"] = list(comp44_all.shape)
    checks["features_log1p_count_shape"] = list(log1p_count.shape)
    checks["features_no_nan"] = bool(np.isfinite(comp44_all).all() and np.isfinite(log1p_count).all())

    source_shas = {
        "trackA_gene_promoters_npz": sha256_file(
            ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_gene_promoters.npz"),
        "split_arrays_npz": split_sha,
        "split_plan_json": sha256_file(DATA05 / "split_plan.json"),
        "split_assignment_tsv": sha256_file(DATA05 / "split_assignment.tsv"),
        "components_tsv": sha256_file(DATA05 / "components.tsv"),
        "label_table_tsv": sha256_file(LABEL_TABLE),
    }

    config = {
        "candidate_version": CANDIDATE,
        "question": "Does a from-scratch gene-bag CNN predicting honest component-fold OOF ridge residuals add predictive value beyond composition/count baselines, on the SAME corrected batch05 split?",
        "scope": "gene-level Score_E ranking only; NOT promoter/construct/HEK293T activity",
        "split": {
            "source": "data/breakthrough_20261002/batch05/split_arrays.npz (frozen, corrected)",
            "sha256": EXPECTED_SPLIT_ARRAYS_SHA,
            "type": "exact/RC + detected-near-identity blocked (NOT homology-free)",
            "n_genes": 57827, "n_train": 46263, "n_val": 5782, "n_test": 5782,
            "n_components": 55402,
        },
        "label": {
            "field": "Score_E",
            "definition": "rank-scaled [0,1] gene-level expression strength; E_raw=mean over tissues of log2(1+TPM); Score_E=rank(E_raw)/(n-1) over canonical 58988-gene table, stable-sort ties",
            "denominator": "canonical 58988-gene label table",
            "no_relabel_from_test": True,
        },
        "features": {
            "comp44": "per-promoter mono(4)+dinuc(16)+GC(1)+length(1) for sym400 and up2000 => 44; gene=mean over promoters",
            "count": "log1p(n_promoters)",
            "total": "45 features (comp44 + log1p count)",
            "aggregation": "gene-mean over promoters",
            "no_label_derived_features": True,
        },
        "outer_folds": {
            "n_folds": N_FOLDS,
            "seed": 20261011,
            "algorithm": "components sorted by gene_count desc, tie-break seeded hash; greedy assign to bin minimizing (count+size)/target_fraction (target 0.2 each)",
            "component_preserving": True,
            "on": "outer TRAIN only",
        },
        "inner_holdout": {
            "target_val_fraction": 0.10,
            "seed": 20261011,
            "algorithm": "same component-preserving greedy assignment",
            "used_for": "ridge alpha selection within each 4-fold subset AND full outer train final ridge selection",
        },
        "ridge": {
            "alpha_grid": ALPHA_GRID,
            "selection_metric": "inner-val Spearman (primary)",
            "tie_rule": "prefer larger alpha",
            "scaling": "z-score using inner-train (selection) / full outer train (final fit) only",
        },
        "baselines": {
            "ridge_comp44_count": "reproduced current ridge comp44+count with nested/inner rules",
            "hgb": "sklearn HistGradientBoostingRegressor(max_iter=200, learning_rate=.1, max_leaf_nodes=15, l2_regularization=.1, early_stopping=False, random_state=20261011), outer train only",
            "existing_batch05_ridge": "batch05 saved comp44+count ridge predictions (matched gene IDs, descriptive)",
            "zero_residual_finalridge": "finalridge prediction (nested-model reference)",
            "b2_original": "batch05 B2 saved predictions (descriptive read-only comparator)",
        },
        "neural": {
            "architecture": "batch03 GeneBagModel: Trunk(conv1 4->64 k7, conv2 64->64 k5, conv3 64->128 k5, BN+ReLU); head Linear(256,1) on concat(mean,max)",
            "arm": "diverse_bag (B2) 4-slot: anchor p0=(epoch+g)%n, slots=(p0+k)%n k=0..3",
            "init": "random init per seed",
            "optimizer": "AdamW lr=0.001 weight_decay=0.0001",
            "batch": 32, "epochs": 12, "final_only": True,
            "target": "MSE residual target = canonical Score_E - OOF ridge prediction (outer train)",
            "seeds": SEEDS,
            "n_runs": 2,
            "no_label_scaling": True,
        },
        "combination": {
            "primary": "finalridge prediction + CNN residual prediction (un-clipped)",
            "fixed_mean": "finalridge + arithmetic mean of the two seed residual outputs",
            "no_blend_weight_clipping_ensemble": True,
        },
        "evaluation": {
            "metrics": ["Spearman", "R2", "MAE", "MSE", "n"],
            "bootstrap": "component-level (whole components resampled, 2000 iters)",
            "strata": ["n=1", "n=2-3", "n>=4", "historical donor unexposed"],
            "strata_thresholds_from_train": True,
            "acceptance": "fixedmean vs EACH simple baseline >= .02 test Spearman gain with bootstrap 95% LB > 0 AND >= 3% relative MAE reduction; individual seeds nonnegative gain",
        },
    }

    all_pass = all(v is not False for v in checks.values() if isinstance(v, bool))
    out = {
        "checks": checks,
        "config": config,
        "source_shas": source_shas,
        "all_pass": all_pass,
        "written_at": None,
    }
    import datetime
    out["written_at"] = datetime.datetime.now().astimezone().isoformat()
    atomic_write_json(REPORT / "preflight_b6.json", out)
    atomic_write_json(REPORT / "frozen_config_b6.json", config)

    import json
    print(json.dumps(checks, indent=2, ensure_ascii=False, default=float))
    print("all_pass =", all_pass)
    if not all_pass:
        print("PREFLIGHT FAILED")
        sys.exit(1)
    print("PREFLIGHT PASS")


if __name__ == "__main__":
    main()
