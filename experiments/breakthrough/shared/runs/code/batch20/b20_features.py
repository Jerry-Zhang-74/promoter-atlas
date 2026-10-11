#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact 1069-feature recomputation from RAW promoter-bag sequence.

Formulas are a line-for-line re-implementation of the frozen upstream source
(read-only) so that features computed from a supplied gene-bag FASTA can be
parity-checked against the frozen catalog matrix:

  44 composition  <- batch03 baseline_seq_ridge.promoter_features
  1024 positional <- batch10 b10_common.window_tri_tetra + b10_extract ordering
  total order     <- batch16 b16_common.FEATURE_NAMES_ALL = comp44 + log1p_count + 1024

Per-promoter features are computed then aggregated by ARITHMETIC MEAN over all
promoters of the gene (matches batch05/batch06/batch10/batch16).
"""
from __future__ import annotations

import numpy as np

from b20_common import AMBIG, BASES, KMERS3, KMERS4, L400, L2000, N_FEATURES  # noqa: E402

BINS = {L400: [0, 100, 200, 300, 400], L2000: [0, 500, 1000, 1500, 2000]}

FEATURE_NAMES_44 = []
for _region in ("sym400", "up2000"):
    for _b in BASES:
        FEATURE_NAMES_44.append("%s_mono_%s" % (_region, _b))
    for _a in BASES:
        for _b in BASES:
            FEATURE_NAMES_44.append("%s_dinuc_%s%s" % (_region, _a, _b))
    FEATURE_NAMES_44.append("%s_gc" % _region)
    FEATURE_NAMES_44.append("%s_len" % _region)

FEATURE_NAMES_1024 = []
for _region in ("sym400", "up2000"):
    for _b in range(4):
        for _k in KMERS3:
            FEATURE_NAMES_1024.append("%s_bin%d_tri_%s" % (_region, _b, _k))
for _region in ("sym400", "up2000"):
    for _k in KMERS4:
        FEATURE_NAMES_1024.append("%s_tetra_%s" % (_region, _k))

FEATURE_NAMES_ALL = FEATURE_NAMES_44 + ["log1p_count"] + FEATURE_NAMES_1024
assert len(FEATURE_NAMES_ALL) == N_FEATURES


def kmer_counts(seg: np.ndarray, k: int):
    """Count valid (all-ACGT) contiguous k-mers per row (exact source replica)."""
    seg = np.asarray(seg, dtype=np.int64)
    n_rows, width = seg.shape
    n = width - k + 1
    idx = np.zeros((n_rows, n), dtype=np.int64)
    valid = np.ones((n_rows, n), dtype=bool)
    for j in range(k):
        win = seg[:, j:j + n]
        valid &= (win < AMBIG)
        idx = idx * 4 + win
    counts = np.zeros((n_rows, 4 ** k), dtype=np.int64)
    if valid.any():
        v = idx[valid]
        ri = np.repeat(np.arange(n_rows), valid.sum(axis=1))
        flat = ri * (4 ** k) + v
        counts = np.bincount(flat, minlength=n_rows * (4 ** k)).reshape(n_rows, 4 ** k)
    return counts, valid.sum(axis=1)


def comp44(seq400: np.ndarray, seq2000: np.ndarray) -> np.ndarray:
    """(n,44) per-promoter composition features (mono/dinuc/GC/length)."""
    feats = []
    for s, L in ((seq400, L400), (seq2000, L2000)):
        s = np.asarray(s, dtype=np.int64)
        n = s.shape[0]
        mono = np.zeros((n, 4), dtype=np.float64)
        for c in range(4):
            mono[:, c] = (s == c).sum(axis=1)
        di = np.zeros((n, 16), dtype=np.float64)
        for a in range(4):
            for b in range(4):
                di[:, a * 4 + b] = ((s[:, :-1] == a) & (s[:, 1:] == b)).sum(axis=1)
        gc = (mono[:, 1] + mono[:, 2]) / L
        length = np.full(n, float(L), dtype=np.float64)
        feats.append(np.concatenate(
            [mono / L, di / (L - 1), gc[:, None], length[:, None]], axis=1))
    return np.concatenate(feats, axis=1)


def _tri_tetra(seq: np.ndarray, L: int):
    n = seq.shape[0]
    edges = BINS[L]
    tri = np.zeros((n, 256), dtype=np.float64)
    col = 0
    for b in range(4):
        seg = seq[:, edges[b]:edges[b + 1]]
        cnt, denom = kmer_counts(seg, 3)
        freq = np.zeros_like(cnt, dtype=np.float64)
        pos = denom > 0
        freq[pos] = cnt[pos] / denom[pos, None]
        tri[:, col:col + 64] = freq
        col += 64
    cnt4, denom4 = kmer_counts(seq, 4)
    tetra = np.zeros_like(cnt4, dtype=np.float64)
    pos4 = denom4 > 0
    tetra[pos4] = cnt4[pos4] / denom4[pos4, None]
    return tri, tetra


def block1024(seq400: np.ndarray, seq2000: np.ndarray) -> np.ndarray:
    """(n,1024) per-promoter positional features, source column order."""
    tri_sym, tetra_sym = _tri_tetra(np.asarray(seq400, dtype=np.int64), L400)
    tri_up, tetra_up = _tri_tetra(np.asarray(seq2000, dtype=np.int64), L2000)
    return np.concatenate([tri_sym, tri_up, tetra_sym, tetra_up], axis=1)


def gene_1069(seq400: np.ndarray, seq2000: np.ndarray) -> np.ndarray:
    """(1069,) gene feature = mean over ALL promoters of (comp44 | log1p | 1024)."""
    seq400 = np.asarray(seq400, dtype=np.int64)
    seq2000 = np.asarray(seq2000, dtype=np.int64)
    if seq400.ndim == 1:
        seq400 = seq400[None, :]
    if seq2000.ndim == 1:
        seq2000 = seq2000[None, :]
    if seq400.shape != (len(seq400), L400) or seq2000.shape != (len(seq2000), L2000):
        raise ValueError("expected (n,400) and (n,2000) sequences")
    if len(seq400) != len(seq2000) or len(seq400) == 0:
        raise ValueError("promoter bag must be non-empty and paired")
    c = comp44(seq400, seq2000).mean(axis=0)
    b = block1024(seq400, seq2000).mean(axis=0)
    log1p_count = np.log1p(float(len(seq400)))
    return np.concatenate([c, [log1p_count], b])
