#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch22 v6.17.0-hektransfer.rc1 shared utilities (NEW, self-contained).

Scope: SMALL frozen-representation transfer / domain-evaluation experiment on the
verified GSE188639 core-promoter library (676 members; 673 with published 133 bp
measured sequences from Haberle et al. 2019 Supplementary Table 6). Target is the
published HEK293 (NOT HEK293T) bulk log2(RNA/input) value `hek_norm`. K562 values
are paired-cell DIAGNOSTIC references only.

Nothing here re-fits, re-selects or overwrites any prior model. All prior batch
sources are read-only. ZERO neural end-to-end fits are authorized in this batch.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch22"
RUN_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch22"
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch22"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch22"
REPORT_ROOT = ROOT / "reports" / "breakthrough_20261002"
REPORT_OUT = REPORT_ROOT / "batch22"

# ---- read-only prior sources -------------------------------------------------
BATCH21_PROBE = ROOT / "data" / "breakthrough_20261002" / "batch21" / "_probe"
BATCH21_MODEL = ROOT / "models" / "breakthrough_20261002" / "batch21"
BATCH21_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch21"
BATCH02_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch02"
FROZEN_CKPT = BATCH21_MODEL / "ckpt_A_seed20261023.pt"   # RecipeA, MSE arm, seed 20261023
FROZEN_FIT_JSON = BATCH21_MODEL / "fit_A_seed20261023.json"

# ---- batch22 acquired sources ------------------------------------------------
SRC = DATA_OUT / "source"
BULK_EXP = SRC / "GSE188639_mixed_cell_bulk_exp.csv.gz"
HABERLE_TABLE = SRC / "haberle2019_table6_human_CP_candidates.tsv"
HABERLE_XLSX = SRC / "nat_m7_human_CP_candidates.xlsx"

VERSION = "v6.17.0-hektransfer.rc1"
BATCH = "batch22"
INSERT_LEN = 133
TARGET_COL = "hek_norm"
DIAG_COL = "k562_norm"

# Deterministic seeds (frozen in the protocol; not score-selected)
SEED_GROUPS = 20261025      # batch22 protocol: group folds seed
SEED_BOOT = 20261025
ALPHA = 100.0               # ridge alpha (fixed)
N_FOLD_5 = 5
N_FOLD_3 = 3
HAM = 16


def set_thread_limits(n: int = 4) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[k] = str(n)


def sha256_file(p) -> str:
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def atomic_write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def now_local() -> str:
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
AMBIG = 4


def rc(s: str) -> str:
    return s.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def encode_onehot(seq: str) -> np.ndarray:
    """Return (4, L) float32 one-hot (ACGT; N/other -> all zero)."""
    L = len(seq)
    x = np.zeros((4, L), dtype=np.float32)
    for j, c in enumerate(seq.upper()):
        v = BASE.get(c)
        if v is not None:
            x[v, j] = 1.0
    return x


# ---------------------------------------------------------------- metrics
def _rank(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(pred, obs) -> float:
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(p) & np.isfinite(o)
    p, o = p[m], o[m]
    if len(p) < 2:
        return float("nan")
    a, b = _rank(p), _rank(o)
    a = a - a.mean(); b = b - b.mean()
    d = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def mae(pred, obs) -> float:
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    return float(np.mean(np.abs(p - o)))


def mse(pred, obs) -> float:
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    return float(np.mean((p - o) ** 2))


def r2(pred, obs) -> float:
    o = np.asarray(obs, float)
    ss = ((o - o.mean()) ** 2).sum()
    return float(1.0 - ((np.asarray(pred, float) - o) ** 2).sum() / ss) if ss > 0 else float("nan")


def top_decile_precision(pred, obs, tie_keys=None) -> dict:
    """k = ceil(0.1*n). Observed top-decile is defined INDEPENDENTLY from the
    measured values. Ties broken by a stable SHA tie key (ascending)."""
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    n = len(p)
    k = int(math.ceil(0.1 * n))
    if tie_keys is None:
        tie_keys = [str(i) for i in range(n)]
    key = np.array([str(s) for s in tie_keys])
    pred_order = np.lexsort((key, -p))
    meas_order = np.lexsort((key, -o))
    pset = set(pred_order[:k].tolist()); mset = set(meas_order[:k].tolist())
    inter = len(pset & mset)
    base = k / n
    return {"k": k, "n": n, "intersection": inter, "precision": inter / k,
            "enrichment": (inter / k) / base if base > 0 else float("nan")}


def group_bootstrap(pred, obs, groups, metric=spearman, n_boot=2000, seed=20261025):
    pred = np.asarray(pred, float); obs = np.asarray(obs, float)
    groups = np.asarray([str(g) for g in groups])
    ug = np.unique(groups)
    if len(ug) < 3:
        return {"point": float(metric(pred, obs)), "lo": None, "hi": None,
                "n_groups": int(len(ug)), "note": "too few independent groups"}
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), size=len(ug))
        idx = np.concatenate([idx_by_g[ug[j]] for j in pick])
        v = metric(pred[idx], obs[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    if len(vals) < 10:
        return {"point": float(metric(pred, obs)), "lo": None, "hi": None,
                "n_groups": int(len(ug)), "note": "bootstrap unstable"}
    return {"point": float(metric(pred, obs)), "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)), "n_groups": int(len(ug)),
            "n_boot": int(len(vals))}
