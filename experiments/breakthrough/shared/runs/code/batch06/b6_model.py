#!/usr/bin/env python3
"""Batch06 model references: re-use reviewed batch03 GeneBagModel read-only.

Identical to corrected batch05 B2 (diverse 4-slot). random init per seed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
sys.path.insert(0, str(BATCH03))

from scratch_gene_model import GeneBagModel, onehot, n_params, Trunk  # noqa: E402,F401
import torch  # noqa: E402

EMB_DIM = 128
N_SLOTS = 4
L400 = 400


def pool_meanmax_4slot(model, x):
    z = model.embed(x)
    B = x.shape[0] // 4
    emb = z.shape[1]
    z = z.reshape(B, 4, emb)
    mean = z.mean(dim=1)
    maxv = z.amax(dim=1)
    return model.head(torch.cat([mean, maxv], dim=1)).squeeze(-1)
