#!/usr/bin/env python3
"""Batch05 data loader: batch03 human-only data + frozen sequence-component split.

The frozen split (exact/RC + detected-near-identity blocked, NOT homology-free)
is read from data/breakthrough_20261002/batch05/split_arrays.npz produced by
b5_split.py BEFORE any heldout score was inspected. The old batch03 `split`
field is intentionally ignored; gene_split here is the new frozen assignment.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
sys.path.insert(0, str(BATCH03))

from train_scratch_gene import make_memmap, gene_order  # noqa: E402,F401

DATA03 = ROOT / "data" / "breakthrough_20261002" / "batch03"
DATA05 = ROOT / "data" / "breakthrough_20261002" / "batch05"
N_HUMAN = 507365
L400 = 400


def load_data_b5():
    z = np.load(DATA03 / "trackA_gene_promoters.npz", allow_pickle=True)
    gene_id = z["gene_id"].astype(str)
    r400 = z["r400"].astype(np.int64)
    score_e = z["Score_E"].astype(np.float64)
    starts = z["starts"].astype(np.int64)
    n_genes = len(starts) - 1

    a = np.load(DATA05 / "split_arrays.npz", allow_pickle=True)
    gene_split = a["gene_split"]            # object array of 'train'/'val'/'test'
    comp_dense = a["comp_dense"].astype(np.int64)
    n_prom = np.diff(starts).astype(np.int64)
    gene_score_e = score_e[starts[:-1]]
    gene_gid = gene_id[starts[:-1]]
    return (gene_id, r400, score_e, starts, gene_split, gene_score_e,
            n_genes, comp_dense, n_prom, gene_gid)


def inner_split_components(comp_dense, train_gene_idx, seed=20261011, target_val=0.10):
    """Deterministic group-preserving inner train/val fold over whole components.

    Mirrors the frozen main assignment algorithm: components sorted by gene count
    desc (seeded tie-break on component id), then greedily assigned to the bin
    minimizing (count+size)/target_fraction. No component is ever split across
    inner-train and inner-val.
    """
    train_comp = comp_dense[train_gene_idx]
    comps = np.unique(train_comp)
    sizes = {int(c): int((train_comp == c).sum()) for c in comps}
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
    inner_val_mask = np.array([assign[int(c)] == "val" for c in train_comp], dtype=bool)
    inner_tr = train_gene_idx[~inner_val_mask]
    inner_val = train_gene_idx[inner_val_mask]
    return inner_tr, inner_val
