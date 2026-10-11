#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gene E/S/B inference using the previously validation-selected rc2 checkpoint.

Architecture and one-hot encoding are imported READ-ONLY from the frozen source
(batch16/b16_model.py) so this layer cannot silently diverge from the trained
network. Selection is inherited: independent arm A, reference seed 20261016
(chosen by the frozen validation rule in batch16; NOT re-selected here).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from b20_common import (BATCH16_CODE, B16_MODEL_A, B16_SCALERS_NPZ, L400, N_SLOTS,
                        CANDIDATE_VERSION, sha256_file)

sys.path.insert(0, str(BATCH16_CODE))
from b16_model import SingleTaskModel  # noqa: E402

TASKS = ("E", "S", "B")
UNITS = {
    "E": "Score_E rank-scaled [0,1] gene-level expression strength",
    "S": "S_raw = -MAD(log2(1+detectable TPM)); defined only for B>=0.30 and >=3 detectable tissues",
    "B": "B_raw = detectable_tissue_count / tissue_count [0,1]",
}


class GeneESBPredictor:
    """Deterministic, CPU-first gene E/S/B predictor (independent arm A)."""

    def __init__(self):
        self.scalers = np.load(B16_SCALERS_NPZ, allow_pickle=True)
        self.models = {}
        self.model_sha = {}
        for t in TASKS:
            p = B16_MODEL_A / ("model_%s.pt" % t)
            m = SingleTaskModel(t)
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"])
            m.eval()
            self.models[t] = m
            self.model_sha[t] = sha256_file(p)
        self.x_mean = self.scalers["X_mean"]
        self.x_std = self.scalers["X_std"]
        self.t_mean = {t: float(self.scalers["%s_mean" % t]) for t in TASKS}
        self.t_std = {t: float(self.scalers["%s_std" % t]) for t in TASKS}

    def predict(self, seqs_u8: np.ndarray, features_1069: np.ndarray, batch: int = 64) -> dict:
        """seqs_u8 (N,4,400) uint8; features (N,1069) float -> dict of (N,) arrays.

        EXACT source layout: one-hots each slot to (4,400); gene i consumes rows
        [i*4:(i+1)*4] of the flattened slot tensor (verified vs the shipped rc2).
        """
        seqs = np.asarray(seqs_u8, dtype=np.uint8)
        feats = np.asarray(features_1069, dtype=np.float64)
        if seqs.ndim == 2 and seqs.shape == (N_SLOTS, L400):
            seqs = seqs[None, :, :]
        if seqs.ndim != 3 or seqs.shape[1:] != (N_SLOTS, L400):
            raise ValueError("expected (N,4,400) uint8, got %s" % (seqs.shape,))
        if feats.ndim == 1:
            feats = feats[None, :]
        if feats.shape != (seqs.shape[0], 1069):
            raise ValueError("features must be (N,1069); got %s vs N=%d" % (feats.shape, seqs.shape[0]))
        if seqs.size == 0 or feats.size == 0:
            raise ValueError("empty batch (N=0) is not accepted")
        if seqs.min() < 0 or seqs.max() > 4:
            raise ValueError("seqs values must be in 0..4 (A,C,G,T,N)")
        if not np.isfinite(feats).all():
            raise ValueError("non-finite feature values")
        xs = (feats - self.x_mean) / self.x_std
        n = seqs.shape[0]
        out = {t: np.empty(n, dtype=np.float64) for t in TASKS}
        with torch.no_grad():
            for i in range(0, n, batch):
                sl = slice(i, min(i + batch, n))
                chunk = seqs[sl]
                m = chunk.shape[0]
                x_seq = onehot_tensor(chunk.reshape(-1, L400))
                x_feat = torch.tensor(xs[sl], dtype=torch.float32)
                for t in TASKS:
                    z = self.models[t](x_seq, x_feat).numpy()
                    out[t][sl] = z * self.t_std[t] + self.t_mean[t]
        return out


def onehot_tensor(flat_u8: np.ndarray) -> torch.Tensor:
    """(N,400) uint8 A/C/G/T/N=0..4 -> (N,4,400) float, N->all-zero (source contract)."""
    x = torch.from_numpy(np.asarray(flat_u8, dtype=np.int64))
    n, l = x.shape
    oh = torch.zeros(n, 4, l, dtype=torch.float32)
    ok = x < 4
    bidx = torch.arange(n).unsqueeze(1).expand(n, l)
    pos = torch.arange(l).unsqueeze(0).expand(n, -1)
    oh[bidx[ok], x[ok], pos[ok]] = 1.0
    return oh
