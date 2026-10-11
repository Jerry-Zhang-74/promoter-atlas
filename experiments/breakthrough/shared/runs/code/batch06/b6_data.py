#!/usr/bin/env python3
"""Batch06 data: frozen batch05 split + features + component-preserving folds.

Re-uses the corrected exact frozen batch05 split arrays (split_arrays.npz),
source sequences/labels, and feature aggregation identical to batch05 ridge.
No graph/split/seed/exclusion changes. Adds a deterministic 5-fold
component-preserving fold over outer TRAIN only (seed 20261011), plus a nested
component-preserving inner holdout for alpha selection within any gene subset.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
BATCH05 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch05"
sys.path.insert(0, str(BATCH03))
sys.path.insert(0, str(BATCH05))

from baseline_seq_ridge import promoter_features  # noqa: E402
from b5_data import load_data_b5  # noqa: E402

DATA03 = ROOT / "data" / "breakthrough_20261002" / "batch03"
L400 = 400
L2000 = 2000
N_HUMAN = 507365


def load_b6():
    """Same frozen batch05 corrected split + source sequences/labels."""
    return load_data_b5()


def make_memmap400():
    return np.memmap(ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap",
                     dtype="uint8", mode="r", shape=(N_HUMAN, L400))


def make_memmap2000():
    return np.memmap(ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap",
                     dtype="uint8", mode="r", shape=(N_HUMAN, L2000))


def load_features():
    """Return comp44_all (n_genes x 44), log1p_count (n_genes,), gc_gene (n_genes,).

    Feature contract (identical to batch05 ridge):
      per-promoter mono(4)+dinuc(16)+GC(1)+length(1) for sym400 and up2000 => 44
      gene feature = mean over the gene's promoters.
      count feature = log1p(n_promoters).
    No output-label-derived features.
    """
    (gene_id, r400, score_e, starts, gene_split, gene_score_e,
     n_genes, comp_dense, n_prom, gene_gid) = load_b6()
    z = np.load(DATA03 / "trackA_gene_promoters.npz", allow_pickle=True)
    r2000 = z["r2000"].astype(np.int64)
    mm400 = make_memmap400()
    mm2000 = make_memmap2000()
    pf = promoter_features(mm400, mm2000, r400, r2000)

    def gene_mean(pf, idx):
        out = np.zeros((len(idx), pf.shape[1]), dtype=np.float64)
        for k, g in enumerate(idx):
            out[k] = pf[int(starts[g]):int(starts[g + 1])].mean(axis=0)
        return out

    comp44_all = gene_mean(pf, np.arange(n_genes))
    log1p_count = np.log1p(n_prom.astype(np.float64))
    gc_gene = comp44_all[:, 20]
    return comp44_all, log1p_count, gc_gene, n_prom


def make_folds_components(comp_dense, gene_idx, n_folds=5, seed=20261011):
    """Deterministic component-preserving n_folds over gene_idx (whole components).

    Components sorted by gene count desc, tie-break = seeded hash of component id;
    greedy assign to the bin minimizing (count+size)/target_fraction. Every fold
    target is 1/n_folds. Returns fold_id array aligned to gene_idx (int 0..n_folds-1)
    and per-fold component assignment for provenance.
    """
    sub = comp_dense[gene_idx]
    comps = np.unique(sub)
    sizes = {int(c): int((sub == c).sum()) for c in comps}
    rng = np.random.default_rng(seed)
    tie = rng.integers(0, 2 ** 31, size=len(comps))
    order = sorted(range(len(comps)),
                   key=lambda i: (-int(sizes[int(comps[i])]), int(tie[i])))
    target = {f: 1.0 / float(n_folds) for f in range(n_folds)}
    bin_counts = {f: 0 for f in range(n_folds)}
    assign = {}
    for i in order:
        c = int(comps[i])
        sz = sizes[c]
        best_bin, best_ratio = None, None
        for f in range(n_folds):
            ratio = (bin_counts[f] + sz) / target[f]
            if best_ratio is None or ratio < best_ratio:
                best_ratio, best_bin = ratio, f
        assign[c] = best_bin
        bin_counts[best_bin] += sz
    fold_id = np.array([assign[int(c)] for c in sub], dtype=np.int64)
    return fold_id, assign


def inner_split_components(comp_dense, gene_idx, seed=20261011, target_val=0.10):
    """Deterministic group-preserving inner train/val over whole components.

    Mirrors the frozen main assignment algorithm. Used inside each 4-fold subset
    for ridge alpha selection, and inside full outer train for the final ridge.
    """
    sub = comp_dense[gene_idx]
    comps = np.unique(sub)
    sizes = {int(c): int((sub == c).sum()) for c in comps}
    rng = np.random.default_rng(seed)
    tie = rng.integers(0, 2 ** 31, size=len(comps))
    order = sorted(range(len(comps)),
                   key=lambda i: (-int(sizes[int(comps[i])]), int(tie[i])))
    target = {"tr": 1.0 - float(target_val), "val": float(target_val)}
    bin_counts = {"tr": 0, "val": 0}
    assign = {}
    for i in order:
        c = int(comps[i])
        sz = sizes[c]
        best_bin, best_ratio = None, None
        for name in ["tr", "val"]:
            ratio = (bin_counts[name] + sz) / target[name]
            if best_ratio is None or ratio < best_ratio:
                best_ratio, best_bin = ratio, name
        assign[c] = best_bin
        bin_counts[best_bin] += sz
    inner_val_mask = np.array([assign[int(c)] == "val" for c in sub], dtype=bool)
    inner_tr = gene_idx[~inner_val_mask]
    inner_val = gene_idx[inner_val_mask]
    return inner_tr, inner_val


def zscore(Xtr, X):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd
