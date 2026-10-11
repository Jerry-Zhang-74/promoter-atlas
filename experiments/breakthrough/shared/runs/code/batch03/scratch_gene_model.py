#!/usr/bin/env python3
"""Batch03 Track A scratch sequence-encoder model definitions (v6.6.0-scratchgene).

A small 1D CNN trunk is shared by two matched-capacity readouts:
  single-promoter (candidate i): per-promoter scalar, gene-mean inference
  gene-bag        (candidate ii): per-gene mean+max embedding pooling -> scalar

Sequences are uint8 over {A,C,G,T,N}=0..4; one-hot encodes ACGT with N -> zero.
Both models are trained from random initialization (no v6.3 weights, no teacher).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

L400 = 400
L2000 = 2000
ALPHABET = 4  # ACGT


def onehot(x: torch.Tensor, length: int = L400) -> torch.Tensor:
    """x: (B, length) long tensor in 0..4 (4=N). Returns (B, 4, length) float."""
    b, l = x.shape
    oh = torch.zeros(b, 4, l, dtype=torch.float32, device=x.device)
    ok = x < 4
    bidx = torch.arange(b, device=x.device).unsqueeze(1).expand(b, l)
    pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, -1)
    oh[bidx[ok], x[ok], pos[ok]] = 1.0
    return oh


class Trunk(nn.Module):
    """Small 1D CNN: (B,4,L)->(B,out_dim,L). Shared capacity for both readouts."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        return h


class SinglePromoterModel(nn.Module):
    """candidate (i): per-promoter scalar; gene score = mean of promoter scalars."""

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.trunk = Trunk(emb_dim)
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)                 # (B, emb_dim, L)
        z = h.mean(dim=2)                 # (B, emb_dim)
        return self.head(z).squeeze(-1)   # (B,)


class GeneBagModel(nn.Module):
    """candidate (ii): per-gene concat(mean, max) over promoter embeddings -> scalar."""

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.trunk = Trunk(emb_dim)
        self.head = nn.Linear(2 * emb_dim, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)                 # (B, emb_dim, L)
        return h.mean(dim=2)              # (B, emb_dim)

    def forward(self, x: torch.Tensor, gene_idx: torch.Tensor, n_genes: int) -> torch.Tensor:
        """x: (T,4,L) flattened promoters. gene_idx: (T,) long gene index 0..n_genes-1."""
        z = self.embed(x)                 # (T, emb_dim)
        emb_dim = z.shape[1]
        # mean over each gene's promoters
        mean = torch.zeros(n_genes, emb_dim, dtype=z.dtype, device=z.device)
        mean.index_add_(0, gene_idx, z)
        counts = torch.bincount(gene_idx, minlength=n_genes).clamp(min=1).unsqueeze(1)
        mean = mean / counts
        # max over each gene's promoters via scatter_reduce amax
        maxv = torch.full((n_genes, emb_dim), float("-inf"), dtype=z.dtype, device=z.device)
        maxv = maxv.scatter_reduce_(0, gene_idx.unsqueeze(1).expand(-1, emb_dim), z,
                                    reduce="amax", include_self=True)
        return self.head(torch.cat([mean, maxv], dim=1)).squeeze(-1)  # (n_genes,)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
