#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch41 v6.28.0-hekft-fragment-benchmark.rc1 shared helpers.

ZERO estimator fit/solve/calibration/gradient/backward/optimizer-step in this batch.
This is a frozen-model transfer DIAGNOSTIC. All new code lives under code/batch41/.
Old assets are read-only.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("E:/promoter_atlas")
BATCH = "batch41"
VERSION = "v6.28.0-hekft-fragment-benchmark.rc1"
CODE = ROOT / "code" / "batch41"
DATA = ROOT / "data" / "breakthrough_20261002" / "batch41"
REPORT = ROOT / "reports" / "breakthrough_20261002" / "batch41"
FIG = REPORT / "figures"

# ---- frozen read-only inputs -------------------------------------------------
DESIGN_GZ = ROOT / "data/breakthrough_20261002/batch40/source/GSM9580234_design_rmIllegalChars.fa.gz"
DESIGN_SHA = "a61f3d7624bdc1cd6066cb0cfb015d482e2994885cf321da9e0bdb7e753aa82b"
DESIGN_BYTES = 429038
JOINED = ROOT / "reports/breakthrough_20261002/batch40/joined_target.tsv"
S5_RAW = ROOT / "data/breakthrough_20261002/batch39/s5_rows_raw.json"
B22 = ROOT / "data/breakthrough_20261002/batch22"
SEQ_TABLE = B22 / "sequence_table.tsv"
FOLD_TABLE = B22 / "fold_assignment.tsv"
GRAM_H1 = ROOT / "data/breakthrough_20261002/batch24/gram_H1_Sglobal.npy"
B24_MODELS = ROOT / "models/breakthrough_20261002/batch24"
PARM_EXT = ROOT / "data/breakthrough_20261002/batch31/_external"
PARM_FROZEN_MANIFEST = ROOT / "reports/breakthrough_20261002/batch35/frozen_assets.json"

# ---- fixed constants (frozen) ------------------------------------------------
N_FOLDS = 5
TOP_FRAC = 0.10
BOOT_N = 2000
BOOT_SEED = 20261025
MIN_COMPONENT = 10
MIN_BOOT_COMPONENTS = 30
PARM_L_MAX = 600
PARM_FILTER = 125
PARM_LOSS = "poisson"
PARM_BATCH = 32

CANON = set("ACGT")
FWD15 = "AGGACCGGATCAACT"
REV15 = "CATTGCGTGAACCGA"


def now_local() -> str:
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def atomic_write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def atomic_write_text(path, text) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    os.replace(tmp, path)


def atomic_write_tsv(path, header, rows) -> None:
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join(str(x) for x in r))
    atomic_write_text(path, "\n".join(lines) + "\n")


def proc_creation_winfiletime() -> str:
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, os.getpid())  # PROCESS_QUERY_LIMITED_INFORMATION
        c = ctypes.c_ulonglong(); e = ctypes.c_ulonglong()
        x = ctypes.c_ulonglong(); y = ctypes.c_ulonglong()
        k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(x), ctypes.byref(y))
        return str(c.value)
    except Exception as ex:  # noqa: BLE001
        return "unavailable:" + repr(ex)


def nvidia_smi() -> str:
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader"], text=True, timeout=20)
        return out.strip()
    except Exception as ex:  # noqa: BLE001
        return "unavailable:" + repr(ex)


# ---- statistics (no fitting; rank correlation only) --------------------------
def _average_ranks(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    ranks = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman rho with average ranks (ties). NaN if <2 or constant input."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if den == 0:
        return float("nan")
    return float((rx * ry).sum() / den)


def deterministic_rank(values, tie_keys):
    """Rank descending (higher=1) with deterministic tie-break on tie_keys ascending.

    Returns integer ranks 1..n (1 = highest value)."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    idx = sorted(range(n), key=lambda i: (-values[i], tie_keys[i]))
    rank = np.empty(n, dtype=np.int64)
    for r, i in enumerate(idx, start=1):
        rank[i] = r
    return rank


def top_set(values, tie_keys, frac=TOP_FRAC):
    n = len(values)
    k = int(np.ceil(frac * n))
    order = sorted(range(n), key=lambda i: (-np.asarray(values)[i], tie_keys[i]))
    if k < 1:
        k = min(1, n)
    return k, set(order[:k])


def set_thread_limits(n: int = 4) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ.setdefault(k, str(n))


def read_joined():
    """Read the frozen joined target table (batch40, read-only)."""
    with open(JOINED, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return rows


def load_design_records():
    """Read the author raw design FASTA (gz) -> ordered (header, seq)."""
    import gzip
    recs = []
    with gzip.open(DESIGN_GZ, "rt", encoding="utf-8", errors="replace") as f:
        nm = None
        ch = []
        for line in f:
            if line.startswith(">"):
                if nm is not None:
                    recs.append((nm, "".join(ch)))
                nm = line[1:].rstrip("\n")
                ch = []
            else:
                ch.append(line.strip())
        if nm is not None:
            recs.append((nm, "".join(ch)))
    return recs


# ---- fast rank metrics (numpy/scipy; no fitting) -----------------------------
def _rankdata(v):
    from scipy.stats import rankdata
    return rankdata(v, method="average")


def spearman_fast(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def top_indices_fast(values, label_ids, frac=TOP_FRAC):
    """Deterministic top set: primary = value descending, tie-break label_ids ascending."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    k = int(np.ceil(frac * n))
    order = np.lexsort((np.asarray(label_ids), -values))
    return k, order[:k]
