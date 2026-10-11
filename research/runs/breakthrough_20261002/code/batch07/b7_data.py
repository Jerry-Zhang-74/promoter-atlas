#!/usr/bin/env python3
"""Batch07 data loader: frozen batch05 split + batch06 features45 + labels + predictions.

Reads the frozen corrected split (split_arrays.npz), the frozen 45-feature matrix
(features45.npz), the canonical gene-level Score_E labels (trackA_gene_promoters.npz)
and the batch06 saved predictor outputs. All arrays share the same gene-level
ordering (n_genes = 57827). Prediction columns are joined by gene_id.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b7_common import DATA05, DATA06, DATA03, REPORT  # noqa: E402

B6_PRED_TSV = REPORT.parent / "batch06" / "predictions" / "pred_v6.7.0-residualgene.rc1.tsv"
BATCH05_RIDGE_PRED = REPORT.parent / "batch05" / "predictions" / "pred_ann_diagnostics_v5.tsv"
BATCH05_B2_PRED = REPORT.parent / "batch05" / "predictions" / "pred_v6.6.0-scratchgene.rc7_s20261011.tsv"


def load_labels_split():
    """Return gene-level labels + split + components + ids aligned (n_genes = 57827)."""
    z = np.load(DATA03 / "trackA_gene_promoters.npz", allow_pickle=True)
    gene_id = z["gene_id"].astype(str)
    score_e = z["Score_E"].astype(np.float64)
    starts = z["starts"].astype(np.int64)
    n_genes = len(starts) - 1

    a = np.load(DATA05 / "split_arrays.npz", allow_pickle=True)
    gene_split = a["gene_split"]            # 'train'/'val'/'test'
    comp_dense = a["comp_dense"].astype(np.int64)
    n_prom = a["n_promoters"].astype(np.int64)

    gene_gid = gene_id[starts[:-1]]
    gene_score_e = score_e[starts[:-1]]
    return gene_gid, gene_score_e, gene_split, comp_dense, n_prom, n_genes


def load_features():
    """Return X45 (n_genes x 45) = comp44 (44) + log1p_count (1), plus gc_gene."""
    f = np.load(DATA06 / "features45.npz")
    comp44 = f["comp44"].astype(np.float64)
    log1p_count = f["log1p_count"].astype(np.float64)
    X45 = np.concatenate([comp44, log1p_count[:, None]], axis=1)
    gc_gene = comp44[:, 20]
    return X45, comp44, log1p_count, gc_gene


def load_saved_predictions():
    """Load batch06 saved predictions (val+test) keyed by gene_id.

    Returns a dict gene_id -> row dict with hgb/finalridge/combined_fixedmean/
    residual_s20261011/residual_s20261012 plus batch05 ridge/B2 (descriptive).
    """
    df = pd.read_csv(B6_PRED_TSV, sep="\t")
    df["gene_id"] = df["gene_id"].astype(str)
    cols = ["hgb", "finalridge", "combined_fixedmean", "residual_s20261011", "residual_s20261012"]
    for c in cols:
        df[c] = df[c].astype(float)
    b6 = {}
    gid_arr = df["gene_id"].to_numpy()
    for c in cols:
        for gid, val in zip(gid_arr, df[c].to_numpy()):
            b6.setdefault(gid, {})[c] = float(val)

    b5r = pd.read_csv(BATCH05_RIDGE_PRED, sep="\t")
    b5r_map = dict(zip(b5r["gene_id"].astype(str), b5r["pred_E_annridge"].astype(float)))
    b5b2 = pd.read_csv(BATCH05_B2_PRED, sep="\t")
    b5b2_map = dict(zip(b5b2["gene_id"].astype(str), b5b2["pred_E_4slot"].astype(float)))
    return b6, b5r_map, b5b2_map


def build_split_rows(split_name):
    """Return aligned arrays for one split (train/val/test): gid, y, comp, n_prom, idx.

    idx = gene-level index (0..n_genes-1) into X45/features/geometry.
    """
    gene_gid, gene_score_e, gene_split, comp_dense, n_prom, n_genes = load_labels_split()
    idx = np.flatnonzero(gene_split == split_name)
    return gene_gid[idx], gene_score_e[idx], comp_dense[idx], n_prom[idx], idx, gene_gid
