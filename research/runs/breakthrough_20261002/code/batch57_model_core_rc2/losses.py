# -*- coding: utf-8 -*-
"""Group-weighted SmoothL1 losses.

Nabs : L = wSmoothL1(pred_z, y_z)
Njoint: L = wSmoothL1(pred_z, y_z) + 0.5 * wSmoothL1((ps-pl)_z, (ys-yl)_z)

Weights are inverse TRAIN group size normalized to mean 1 (activity) and inverse TRAIN
pair-group size normalized mean 1 (difference). The pair term has NO free intercept: the
difference is formed from the SAME two endpoint heads, so the arm stays coherent.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from spec import NJOINT_PAIR_WEIGHT


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor,
                       weight: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    w = weight.to(per.dtype)
    return (per * w).sum() / w.sum().clamp(min=1e-12)


def activity_loss(pred_z, y_z, weight):
    return weighted_smooth_l1(pred_z, y_z, weight)


def pair_difference_loss(pred_short_z, pred_long_z, y_short_z, y_long_z, pair_weight):
    d_pred = pred_short_z - pred_long_z          # zero pair intercept by construction
    d_true = y_short_z - y_long_z
    return weighted_smooth_l1(d_pred, d_true, pair_weight)


def total_loss(pred_z, y_z, act_weight, pair_weight=None,
               pair_pred=None, pair_true=None, joint: bool = False):
    loss = activity_loss(pred_z, y_z, act_weight)
    parts = {"activity": float(loss.detach())}
    if joint:
        if pair_pred is None or pair_true is None or pair_weight is None:
            raise ValueError("Njoint requires pair prediction, target and weights")
        pl = pair_difference_loss(pair_pred[0], pair_pred[1], pair_true[0], pair_true[1], pair_weight)
        parts["pair"] = float(pl.detach())
        loss = loss + NJOINT_PAIR_WEIGHT * pl
    return loss, parts
