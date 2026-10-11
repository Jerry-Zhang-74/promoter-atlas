#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch23 v6.17.1-sequence-evidence.rc1 shared utilities (NEW, self-contained).

Scope (parent protocol 2026-10-03 15:42):
  1. LOCAL SEQUENCE / ENDPOINT RECONCILIATION -- correct batch22's factual error
     ("no DNA sequence for the 14 local constructs") using the real project records
     (reports/cellline_v2_20261001/construct_sequences.tsv/.fasta,
      construct_sequence_provenance.json, construct_endpoints_D28.tsv,
      reports/breakthrough_20261002/batch04/cellline_14construct_diagnostic_table.tsv,
      data/breakthrough_20261002/batch21/construct_audit.tsv).
  2. UNTUNED LOCAL DIAGNOSTIC -- apply the FIVE SAVED batch22 fold readouts to the
     ACTUAL native full inserts (no truncation, no padding, no reference filling,
     no retraining). Exploratory only; NOT a reliability or activity claim.
  3. PUBLIC DATA ACQUISITION / TRAINING READINESS -- verified source inventory.

Hard limits: ZERO neural training, ZERO new classical estimator fits. Frozen encoder
inference on cuda:0 only with parameter/input assertions. All prior batch sources are
READ-ONLY. Only batch23 directories are written.
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

CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch23"
RUN_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch23"
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch23"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch23"
REPORT_ROOT = ROOT / "reports" / "breakthrough_20261002"
REPORT_OUT = REPORT_ROOT / "batch23"

# ---- read-only prior sources (never modified) --------------------------------
CELLLINE = ROOT / "reports" / "cellline_v2_20261001"
CELLLINE_V1 = ROOT / "reports" / "cellline_v1_20261001"
CONSTRUCT_SEQ_TSV = CELLLINE / "construct_sequences.tsv"
CONSTRUCT_SEQ_FASTA = CELLLINE / "construct_sequences.fasta"
CONSTRUCT_PROV = CELLLINE / "construct_sequence_provenance.json"
ENDPOINTS_D28 = CELLLINE / "construct_endpoints_D28.tsv"
CV_PREDS = CELLLINE / "model_cv_predictions.tsv"
CV_FOLDS = CELLLINE / "model_cv_folds.json"
CELLLINE_METRICS = CELLLINE / "model_metrics.tsv"
EXCLUDED_81 = CELLLINE / "excluded_untransfected_fragment.tsv"

BATCH04_TABLE = REPORT_ROOT / "batch04" / "cellline_14construct_diagnostic_table.tsv"
BATCH21_AUDIT_TSV = ROOT / "data" / "breakthrough_20261002" / "batch21" / "construct_audit.tsv"
BATCH21_AUDIT_JSON = ROOT / "data" / "breakthrough_20261002" / "batch21" / "construct_audit.json"
BATCH20_EXAMPLES = ROOT / "data" / "breakthrough_20261002" / "batch20" / "examples"

BATCH22 = ROOT / "reports" / "breakthrough_20261002" / "batch22"
BATCH22_DATA = ROOT / "data" / "breakthrough_20261002" / "batch22"
BATCH22_MODEL = ROOT / "models" / "breakthrough_20261002" / "batch22"
BATCH22_SEQ_TABLE = BATCH22_DATA / "sequence_table.tsv"
BATCH22_FOLD_ASSIGN = BATCH22_DATA / "fold_assignment.tsv"

BATCH21_MODEL = ROOT / "models" / "breakthrough_20261002" / "batch21"
FROZEN_CKPT = BATCH21_MODEL / "ckpt_A_seed20261023.pt"   # RecipeA seed 20261023 (read-only)

VERSION = "v6.17.1-sequence-evidence.rc1"
BATCH = "batch23"
TRAIN_LEN = 133            # batch22 trained insert length
N_FOLDS = 5
ALPHA = 100.0

# batch22 fig3 hardcoded (WRONG) construct lengths -- preserved as the historical claim
BATCH22_HARDCODED_LENGTHS = [100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 700, 700]


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
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def now_local() -> str:
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


def rc(s: str) -> str:
    return s.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def encode_onehot(seq: str) -> np.ndarray:
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


def pearson(pred, obs) -> float:
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(p) & np.isfinite(o)
    p, o = p[m], o[m]
    if len(p) < 3:
        return float("nan")
    p = p - p.mean(); o = o - o.mean()
    d = math.sqrt((p * p).sum() * (o * o).sum())
    return float((p * o).sum() / d) if d > 0 else float("nan")


def mae(pred, obs) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(obs, float))))


def r2(pred, obs) -> float:
    o = np.asarray(obs, float)
    ss = ((o - o.mean()) ** 2).sum()
    return float(1.0 - ((np.asarray(pred, float) - o) ** 2).sum() / ss) if ss > 0 else float("nan")
