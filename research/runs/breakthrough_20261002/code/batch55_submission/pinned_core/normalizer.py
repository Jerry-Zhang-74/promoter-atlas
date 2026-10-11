# -*- coding: utf-8 -*-
"""TRAIN-only target normalization, frozen once. y_z=(activity-mean)/std, inverse on output.

batch52 PROTOCOL: target normalization uses TRAIN weighted mean/std (weights = inverse
TRAIN group size, normalized mean 1). The normalizer is fitted ONCE on TRAIN and frozen;
DEV/TEST never enter the normalizer.
"""
from __future__ import annotations
import torch


class TargetNormalizer:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std if std != 0 else 1.0)
        self.frozen = False

    @classmethod
    def fit(cls, y: torch.Tensor, weights: torch.Tensor) -> "TargetNormalizer":
        y = y.to(torch.float64)
        w = weights.to(torch.float64)
        w = w / w.mean().clamp(min=1e-12)
        mean = float((w * y).sum() / w.sum())
        var = float((w * (y - mean) ** 2).sum() / w.sum())
        n = cls(mean, var ** 0.5 if var > 0 else 1.0)
        n.frozen = True
        return n

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std, "frozen": self.frozen}
