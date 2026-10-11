# -*- coding: utf-8 -*-
"""R0 composition ridge baseline (batch52 PROTOCOL).

Features: mono(4)+di(16)+tri(64)+tetra(256)=340 kmer frequencies + GC fraction + loglength
= 342, standardized on TRAIN only. Weighted ridge alpha=100 with TRAIN weights = inverse
TRAIN group size normalized mean 1; intercept = weighted TRAIN target mean (scale 1, raw
author log2 units). ONE solve. No pair-specific second ridge: endpoint predictions are
differenced (short - long) to form the pair delta. The helper never runs an actual fit.
"""
from __future__ import annotations
import itertools

import numpy as np

from spec import R0_ALPHA, R0_SCALE

_K = (1, 2, 3, 4)
_KIDX = {}
_i = 0
for k in _K:
    for tup in itertools.product("ACGT", repeat=k):
        _KIDX[(k, "".join(tup))] = _i
        _i += 1
N_KMER = _i                      # 340
N_FEAT = N_KMER + 2              # 342


def _codes_to_seq(codes_row: np.ndarray, length: int) -> str:
    b = "ACGT"
    return "".join(b[c] if c < 4 else "N" for c in codes_row[:length].tolist())


def features_from_codes(codes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """codes: (N,<=600) base codes (0..3, >=4 pad/ambiguous). Returns (N,342) float64."""
    N = codes.shape[0]
    X = np.zeros((N, N_FEAT), dtype=np.float64)
    for i in range(N):
        L = int(lengths[i])
        seq = _codes_to_seq(codes[i], L)
        for k in _K:
            n = L - k + 1
            if n <= 0:
                continue
            for j in range(n):
                sub = seq[j:j + k]
                if "N" in sub:
                    continue
                X[i, _KIDX[(k, sub)]] += 1.0 / n
        gc = sum(1 for c in seq if c in "GC") / max(L, 1)
        X[i, N_KMER] = gc
        X[i, N_KMER + 1] = np.log(max(L, 1))
    return X


def _all_k(k):
    return ["".join(t) for t in itertools.product("ACGT", repeat=k)]


class R0Model:
    def __init__(self):
        self.feat_mean = None
        self.feat_std = None
        self.beta = None
        self.intercept = None
        self.alpha = R0_ALPHA

    def _standardize_fit(self, X):
        self.feat_mean = X.mean(axis=0)
        s = X.std(axis=0)
        s[s == 0] = 1.0
        self.feat_std = s
        return (X - self.feat_mean) / self.feat_std

    def _standardize(self, X):
        return (X - self.feat_mean) / self.feat_std

    def fit(self, X, y, weights):
        """One weighted ridge solve. y raw log2 units, weights normalized mean 1."""
        Z = self._standardize_fit(X)
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.mean()
        self.intercept = float(np.sum(w * y) / np.sum(w))
        resid = y - self.intercept
        XtW = Z.T * w
        A = XtW @ Z + self.alpha * np.eye(Z.shape[1])
        b = XtW @ resid
        self.beta = np.linalg.solve(A, b)
        return self

    def predict(self, X):
        Z = self._standardize(X)
        return (Z @ self.beta) + self.intercept

    def state(self):
        return {"alpha": self.alpha, "scale": R0_SCALE, "feat_mean": self.feat_mean.tolist(),
                "feat_std": self.feat_std.tolist(), "beta": self.beta.tolist(),
                "intercept": self.intercept, "n_features": N_FEAT, "feature_order": "mono,di,tri,tetra,gc,loglen"}
