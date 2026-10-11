#!/usr/bin/env python3
"""Batch02 shared utilities (UTF-8 explicit, deterministic)."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]  # E:/promoter_atlas


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def atomic_write_json(path, obj, sort_keys=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=True, default=float, sort_keys=sort_keys)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = rankdata(a) - np.mean(rankdata(a))
    rb = rankdata(b) - np.mean(rankdata(b))
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def r2(pred, target):
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae(pred, target):
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(target, dtype=float))))


def mse(pred, target):
    return float(np.mean((np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)) ** 2))


def set_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_paired_spearman(y, pred_c, pred_b, n=10000, seed=20261006):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    pred_c = np.asarray(pred_c, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    diffs = []
    for _ in range(n):
        ii = rng.integers(0, len(y), size=len(y))
        sc = spearman(y[ii], pred_c[ii])
        sb = spearman(y[ii], pred_b[ii])
        diffs.append(sc - sb)
    diffs = np.asarray(diffs)
    return {
        "mean_diff": float(np.mean(diffs)),
        "pct_positive": float(np.mean(diffs > 0)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }
