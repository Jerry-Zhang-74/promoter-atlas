# -*- coding: utf-8 -*-
"""batch30 FROZEN architecture, UNCHANGED (no architecture search).
Provenance: DilatedResBlock / LegNetInspired / build_cnn copied VERBATIM from
  runs/breakthrough_20261002/code/batch02/trackB_grouped_cv.py
  sha256 7de6cca87b2868911f9e3148fba0f3a0192d5dc386fc376f30cf5b802196b731 (READ-ONLY).
LegNetInspired(1, ch=64) has 110145 params.  Input codes (B,L) values 0..3 ONLY (no padding code 4).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResBlock(nn.Module):
    def __init__(self, ch, dilation, k=5):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 1)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class LegNetInspired(nn.Module):
    def __init__(self, num_targets=1, ch=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([DilatedResBlock(ch, d) for d in [1, 2, 4, 8]])
        self.head = nn.Sequential(nn.Linear(ch * 2, 64), nn.ReLU(inplace=True), nn.Dropout(0.1),
                                  nn.Linear(64, num_targets))

    def forward(self, x_idx):
        x = F.one_hot(x_idx.long(), num_classes=4).float().permute(0, 2, 1).contiguous()
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        h = torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1)
        return self.head(h)


def build_cnn(name):
    if name == "legnet":
        return LegNetInspired(1)
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
