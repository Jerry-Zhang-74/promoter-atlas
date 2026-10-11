#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch16 neural model definitions (v6.15.0-multitask.rc1).

Identical input representation across arms A/B/C:
  * sequence branch: small 1D CNN trunk over the diverse 4-slot gene-bag
    (one-hot ACGT, N->zero), concat(mean,max) over the 4 slots.
  * feature branch: frozen 1069-feature composition/position branch -> 128.

  A (independent): 3 separate single-head models (each owns seq+feat backbone).
  B (shared)      : one model with E/S/B heads.
  C (shared+aux)  : B plus a promoter-level auxiliary head (7 observed epigenetic
                    fields) attached BEFORE gene pooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

L400 = 400


def onehot(x: torch.Tensor, length: int = L400) -> torch.Tensor:
    """x: (B, length) long in 0..4 (4=N). Returns (B, 4, length) float, N->zero."""
    b, l = x.shape
    oh = torch.zeros(b, 4, l, dtype=torch.float32, device=x.device)
    ok = x < 4
    bidx = torch.arange(b, device=x.device).unsqueeze(1).expand(b, l)
    pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, -1)
    oh[bidx[ok], x[ok], pos[ok]] = 1.0
    return oh


class Trunk(nn.Module):
    """Small 1D CNN (batch03 Trunk): (B,4,L)->(B,128,L)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        return h


class FeatureBranch(nn.Module):
    def __init__(self, n_feat: int = 1069, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(n_feat, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class _SharedRep(nn.Module):
    """Sequence trunk + feature branch -> combined representation."""

    def __init__(self, n_feat: int = 1069, emb: int = 128, feat_hidden: int = 128):
        super().__init__()
        self.trunk = Trunk(emb)
        self.feat = FeatureBranch(n_feat, feat_hidden)
        self.emb = emb
        self.feat_hidden = feat_hidden

    def embed_slots(self, x_seq):
        """x_seq: (B*4, 4, L400) -> per-slot embeddings (B, 4, emb)."""
        z = self.trunk(x_seq).mean(dim=2)          # (B*4, emb)
        B = x_seq.shape[0] // 4
        return z.reshape(B, 4, self.emb)

    def rep(self, x_seq, x_feat):
        z = self.embed_slots(x_seq)                # (B, 4, emb)
        seq_emb = torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=1)  # (B, 2*emb)
        feat_emb = self.feat(x_feat)               # (B, feat_hidden)
        return torch.cat([seq_emb, feat_emb], dim=1), z  # rep (B, 2*emb+feat_hidden), z


class MultiTaskModel(nn.Module):
    """B (shared E/S/B heads) and C (same + promoter-level aux head)."""

    def __init__(self, n_feat: int = 1069, emb: int = 128, feat_hidden: int = 128,
                 n_aux: int = 0):
        super().__init__()
        self.shared = _SharedRep(n_feat, emb, feat_hidden)
        rep_dim = 2 * emb + feat_hidden
        self.head_e = nn.Linear(rep_dim, 1)
        self.head_s = nn.Linear(rep_dim, 1)
        self.head_b = nn.Linear(rep_dim, 1)
        self.aux = nn.Linear(emb, n_aux) if n_aux > 0 else None

    def forward(self, x_seq, x_feat):
        rep, z = self.shared.rep(x_seq, x_feat)
        out = {
            "E": self.head_e(rep).squeeze(-1),
            "S": self.head_s(rep).squeeze(-1),
            "B": self.head_b(rep).squeeze(-1),
        }
        if self.aux is not None:
            out["aux"] = self.aux(z)               # (B, 4, n_aux)
        return out


class SingleTaskModel(nn.Module):
    """A independent single-head model (own seq+feat backbone)."""

    TASKS = ("E", "S", "B")

    def __init__(self, task: str, n_feat: int = 1069, emb: int = 128, feat_hidden: int = 128):
        super().__init__()
        assert task in self.TASKS
        self.task = task
        self.shared = _SharedRep(n_feat, emb, feat_hidden)
        rep_dim = 2 * emb + feat_hidden
        self.head = nn.Linear(rep_dim, 1)

    def forward(self, x_seq, x_feat):
        rep, _ = self.shared.rep(x_seq, x_feat)
        return self.head(rep).squeeze(-1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
