# -*- coding: utf-8 -*-
"""Frozen sequence CNN used IDENTICALLY by Nabs (absolute) and Njoint (pair).

Architecture is transcribed from batch52/PROTOCOL.txt:
  one-hot actual ACGT, zero padding to 600, explicit valid-position mask;
  stem Conv1d(4,64,k9,same)+GELU; THREE residual blocks, each two Conv1d(64,64,k5,same)
  with dilation 1/2/4, GELU, dropout 0.1; mask padded positions after stem and each block;
  masked mean AND masked max pooled, concatenated with standardized loglength/GC;
  linear 130->64, GELU, dropout 0.1 -> scalar.
Masked max must never select padding. Inputs >600 or null are rejected before forward.
"""
from __future__ import annotations
import hashlib, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from spec import (SEED, IN_CH, STEM_CH, STEM_K, BLOCK_CH, BLOCK_K, BLOCK_DILATIONS,
                  N_RES_BLOCKS, DROPOUT, HEAD_HIDDEN, POOL_DIM, MAX_LEN, PAD_CODE)

NEG_INF = -1.0e9


# ---------------------------------------------------------------- encoding ---
def encode_bases(codes: torch.Tensor) -> tuple:
    """codes: LongTensor (N,L) with 0=A 1=C 2=G 3=T and >=4 = padding/ambiguous.

    Returns (onehot (N,4,L) float, mask (N,L) bool). Ambiguous/padding -> zero channels,
    mask False (padding and ambiguous bases are never pooled)."""
    if codes.dim() != 2:
        raise ValueError("codes must be (N,L)")
    L = codes.shape[1]
    if L > MAX_LEN:
        raise ValueError("unsupported length %d > %d (no windowing)" % (L, MAX_LEN))
    if codes.numel() and (codes < 0).any():
        raise ValueError("null/invalid base code")
    valid = codes < 4
    oh = F.one_hot(codes.clamp(0, 3), IN_CH).to(torch.float32) * valid.unsqueeze(-1).to(torch.float32)
    return oh.transpose(1, 2).contiguous(), valid


class PoolStats(nn.Module):
    """TRAIN-only standardizer for the two auxiliary pooled features (loglength, GC)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("loglen_mean", torch.zeros(1))
        self.register_buffer("loglen_std", torch.ones(1))
        self.register_buffer("gc_mean", torch.zeros(1))
        self.register_buffer("gc_std", torch.ones(1))
        self.frozen = False

    @torch.no_grad()
    def fit(self, lengths: torch.Tensor, gc: torch.Tensor):
        self.loglen_mean.copy_(lengths.mean())
        s = lengths.std(unbiased=False)
        self.loglen_std.copy_(s if float(s) > 0 else torch.ones_like(s))
        self.gc_mean.copy_(gc.mean())
        s2 = gc.std(unbiased=False)
        self.gc_std.copy_(s2 if float(s2) > 0 else torch.ones_like(s2))
        self.frozen = True

    def transform(self, lengths: torch.Tensor, gc: torch.Tensor):
        return ((lengths - self.loglen_mean) / self.loglen_std,
                (gc - self.gc_mean) / self.gc_std)


class ResBlock(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        pad = dilation * (BLOCK_K // 2)
        self.c1 = nn.Conv1d(BLOCK_CH, BLOCK_CH, BLOCK_K, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(BLOCK_CH, BLOCK_CH, BLOCK_K, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(1).to(x.dtype)
        y = self.drop(F.gelu(self.c1(x))) * m    # mask so conv bias at padded taps cannot leak
        y = self.drop(F.gelu(self.c2(y))) * m
        y = x + y
        return y * m


class SeqCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(IN_CH, STEM_CH, STEM_K, padding=STEM_K // 2)
        self.blocks = nn.ModuleList([ResBlock(d) for d in BLOCK_DILATIONS])
        self.pool_stats = PoolStats()
        self.head1 = nn.Linear(POOL_DIM, HEAD_HIDDEN)
        self.head_drop = nn.Dropout(DROPOUT)
        self.head2 = nn.Linear(HEAD_HIDDEN, 1)

    def forward(self, codes: torch.Tensor, gc: torch.Tensor = None) -> torch.Tensor:
        oh, mask = encode_bases(codes)
        m = mask.unsqueeze(1).to(oh.dtype)
        x = F.gelu(self.stem(oh)) * m
        for blk in self.blocks:
            x = blk(x, mask)
            x = x * m
        neg = torch.full_like(x, NEG_INF)
        masked = torch.where(mask.unsqueeze(1), x, neg)
        pmax = masked.max(dim=2).values
        valid_any = mask.any(dim=1, keepdim=True)
        pmax = torch.where(valid_any, pmax, torch.zeros_like(pmax))
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        pmean = (x * m).sum(dim=2) / denom
        lengths = mask.sum(dim=1).to(x.dtype)
        if gc is None:
            # GC fraction over valid positions from the one-hot channels C,G.
            cg = (oh[:, 1] + oh[:, 2]) * mask.to(oh.dtype)
            gc = cg.sum(dim=1) / denom.squeeze(1)
        loglen = torch.log(lengths.clamp(min=1.0))
        zl, zg = self.pool_stats.transform(loglen, gc)
        feat = torch.cat([pmean, pmax, zl.unsqueeze(1), zg.unsqueeze(1)], dim=1)
        if feat.shape[1] != POOL_DIM:
            raise AssertionError("pool dim %d != %d" % (feat.shape[1], POOL_DIM))
        h = self.head_drop(F.gelu(self.head1(feat)))
        return self.head2(h).squeeze(1)


# --------------------------------------------------------------- init bytes --
def _seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(seed: int = SEED) -> SeqCNN:
    """Construct a FRESH SeqCNN with deterministic init. Calling twice yields identical
    parameter bytes (verified by param_sha256) -- the shared-init requirement."""
    _seed_all(seed)
    m = SeqCNN()
    return m


def param_sha256(model: nn.Module) -> str:
    """SHA256 over sorted (name, dtype, raw bytes) of all *parameters* (the 'init bytes')."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(str(tuple(p.shape)).encode("ascii"))
        h.update(str(p.dtype).encode("ascii"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assert_same_init(a: nn.Module, b: nn.Module) -> str:
    sa, sb = param_sha256(a), param_sha256(b)
    if sa != sb:
        raise AssertionError("init bytes differ: %s != %s" % (sa, sb))
    return sa


def assert_cuda_all(model: nn.Module, x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise AssertionError("input not on CUDA")
    for n, p in model.named_parameters():
        if not p.is_cuda:
            raise AssertionError("parameter %s not on CUDA" % n)


# ----------------------------------------------------- standalone predictor --
def predict_activity(model: nn.Module, codes: torch.Tensor, normalizer, device="cpu") -> torch.Tensor:
    """Zero-fit forward returning author-unit activity (denormalized). normalizer is a
    normalizer.TargetNormalizer (or object exposing mean/std). No gradients, no training."""
    model.eval()
    with torch.no_grad():
        z = model(codes.to(device))
        return normalizer.inverse(z)
