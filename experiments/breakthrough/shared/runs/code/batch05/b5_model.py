#!/usr/bin/env python3
"""Batch05 model references (copied from verified batch04 b4_model, paths updated).

Both arms B1 (repeat_bag) and B2 (diverse_bag) use the EXACT batch03
GeneBagModel: identical Trunk (conv1 4->64 k7, conv2 64->64 k5, conv3 64->128
k5, BN+ReLU) and identical head Linear(2*128,1) with concat(mean,max) pooling.
Only the per-gene bag construction differs. Importing the reviewed class
read-only guarantees architecture identity.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
BATCH03 = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"
sys.path.insert(0, str(BATCH03))

from scratch_gene_model import GeneBagModel, onehot, n_params, Trunk  # noqa: E402,F401

EMB_DIM = 128
N_SLOTS = 4  # both arms always use exactly 4 active input slots
L400 = 400

ARMS = ("repeat_bag", "diverse_bag")


def pool_meanmax_4slot(model, x):
    """Deterministic mean+max pooling over exactly 4 slots per gene.

    x: (B*4, 4, L400) one-hot, gene-major then slot-minor order.
    Returns head output of shape (B,). Mathematically identical to the batch03
    GeneBagModel concat(mean,max) when every gene has 4 slots, but implemented
    with reshape/reduction (no CUDA atomics), so training/resume is bitwise
    deterministic.
    """
    z = model.embed(x)                 # (B*4, emb)
    B = x.shape[0] // 4
    emb = z.shape[1]
    z = z.reshape(B, 4, emb)
    mean = z.mean(dim=1)               # (B, emb)
    maxv = z.amax(dim=1)               # (B, emb)
    return model.head(torch.cat([mean, maxv], dim=1)).squeeze(-1)
