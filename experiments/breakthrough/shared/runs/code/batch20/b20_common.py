#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch20 v6.16.0-sequence.rc1 shared utilities.

RAW-SEQUENCE-FIRST usability layer: exact preprocessing for the previously
validation-selected sequence/MPRA checkpoint and the gene E/S/B checkpoint,
plus traceability, hashing and UTF-8/thread contracts.

Nothing here refits or reselects a model. All model selection is inherited from
prior batches; this module only implements inference and honest scope labelling.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch20"
BATCH16_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch16"
BATCH02_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch02"
BATCH10_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch10"
BATCH03_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch03"

DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch20"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch20"
REPORT_ROOT = ROOT / "reports" / "breakthrough_20261002"
REPORT_OUT = REPORT_ROOT / "batch20"
RUNS_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch20"

# ---- frozen upstream sources (read-only) ----
TRACKA_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch03" / "trackA_gene_promoters.npz"
B16_DATA_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch16" / "data.npz"
B16_LABELS_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch16" / "labels.npz"
B16_SCALERS_NPZ = ROOT / "data" / "breakthrough_20261002" / "batch16" / "scalers.npz"
B16_MODEL_DIR = ROOT / "models" / "breakthrough_20261002" / "batch16"
B16_MODEL_A = B16_MODEL_DIR / "A_s20261016"
MEM400 = ROOT / "promoters" / "memmap" / "promoter_sym400_uint8.memmap"
MEM2000 = ROOT / "promoters" / "memmap" / "promoter_up2000_uint8.memmap"
ANALYSIS_SET = ROOT / "labels" / "build" / "promoter_analysis_set.tsv"
LABEL_TABLE = ROOT / "labels" / "build" / "gene_expression_label_table.tsv"
MPRA_SUPERVISED = ROOT / "features" / "mpra" / "mpra_supervised.tsv"
MPRA_WEIGHT_DIR = ROOT / "models" / "breakthrough_20261002" / "batch02"
MPRA_WEIGHTS = {c: [MPRA_WEIGHT_DIR / ("weights_%s_fold%d_seed%d.pt" % (c, f, s))
                    for f in range(5) for s in (20261008, 20261009, 20261010)]
                for c in ("legnet", "smallcnn")}

N_HUMAN = 507365
L400 = 400
L2000 = 2000
N_FEATURE_UNIVERSE = 57827
N_FEATURES = 1069
N_SLOTS = 4
MPRA_LEN = 164

CANDIDATE_VERSION = "v6.16.0-sequence.rc1"
BATCH = "batch20"

BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
AMBIG = 4
BASES = "ACGT"
KMERS3 = [a + b + c for a in BASES for b in BASES for c in BASES]
KMERS4 = [a + b + c + d for a in BASES for b in BASES for c in BASES for d in BASES]


class B20InputError(ValueError):
    """Typed user-input error: bad base, length, empty record, ambiguous gene id."""


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


def encode_uint8(seq: str, length: int | None = None, name: str = "") -> np.ndarray:
    """Encode a DNA string to uint8 (A=0,C=1,G=2,T=3,N/other=4).

    EXACT source contract (batch10/batch02): uppercase, per-position mapping,
    any non-ACGT (incl. N and IUPAC ambiguity) -> 4. No silent truncation: if
    `length` is given and does not match, raise B20InputError.
    """
    s = (seq or "").strip().upper()
    if len(s) == 0:
        raise B20InputError("empty sequence%s" % (" for " + name if name else ""))
    if length is not None and len(s) != length:
        raise B20InputError("length %d != required %d%s"
                            % (len(s), length, " for " + name if name else ""))
    out = np.full(len(s), AMBIG, dtype=np.uint8)
    for i, c in enumerate(s):
        v = BASE.get(c)
        if v is not None:
            out[i] = v
    return out


def revcomp_str(s: str) -> str:
    return s.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def read_fasta(path) -> list:
    """Minimal strict FASTA reader -> [(name, sequence)]. Raises on empty/no records."""
    recs = []
    name = None
    buf = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(buf)))
                name = line[1:].strip() or ("seq%d" % (len(recs) + 1))
                buf = []
            else:
                buf.append(line)
    if name is not None:
        recs.append((name, "".join(buf)))
    if not recs:
        raise B20InputError("no FASTA records found in %s" % path)
    return recs


def rmse_mae_spearman(pred, obs):
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if len(pred) < 2:
        return float("nan"), float("nan"), float("nan")
    mae = float(np.mean(np.abs(pred - obs)))
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    rp = np.argsort(np.argsort(pred, kind="mergesort"), kind="mergesort").astype(float)
    ro = np.argsort(np.argsort(obs, kind="mergesort"), kind="mergesort").astype(float)
    rp -= rp.mean()
    ro -= ro.mean()
    d = math.sqrt((rp ** 2).sum() * (ro ** 2).sum())
    return rmse, mae, float((rp * ro).sum() / d) if d else float("nan")
