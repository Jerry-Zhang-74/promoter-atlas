#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch21 v6.16.1-sequence.rc1 shared utilities (NEW, self-contained).

Scope: new-sequence performance experiment on the local K562 GSE118242 promoter
fragment MPRA (in-domain) because no HEK293/HEK293T promoter-reporter source with
processed per-sequence scalar activity was obtainable inside the declared
preparation bounds. The HEK293T gap is recorded explicitly and never bridged by
a K562 score.

Holds paths, hashing, IO, metrics and the raw-input quality rules required by the
batch21 protocol: source-catalog in-source marking, honest fold-aware OOF,
greater-than-5-percent N rejection, and clear empty/noheader/IUPAC/unknown handling.

Nothing here refits or reselects any prior model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

CODE_OUT = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch21"
RUN_OUT = ROOT / "runs" / "breakthrough_20261002" / "batch21"
DATA_OUT = ROOT / "data" / "breakthrough_20261002" / "batch21"
MODEL_OUT = ROOT / "models" / "breakthrough_20261002" / "batch21"
REPORT_ROOT = ROOT / "reports" / "breakthrough_20261002"
REPORT_OUT = REPORT_ROOT / "batch21"

BATCH02_CODE = ROOT / "runs" / "breakthrough_20261002" / "code" / "batch02"
COMPONENTS = ROOT / "data" / "breakthrough_20261002" / "batch02" / "trackB_train_components.tsv"
FOLD_ASSIGN = ROOT / "data" / "breakthrough_20261002" / "batch02" / "trackB_cv_fold_assignment.tsv"
MPRA_SUPERVISED = ROOT / "features" / "mpra" / "mpra_supervised.tsv"
MPRA_FAMILIES = ROOT / "features" / "mpra" / "mpra_families.tsv"
MPRA_WEIGHT_DIR = ROOT / "models" / "breakthrough_20261002" / "batch02"
CELLLINE_V2 = ROOT / "reports" / "cellline_v2_20261001"
CONSTRUCT_ENDPOINTS = CELLLINE_V2 / "construct_endpoints_D28.tsv"
CONSTRUCT_SEQUENCES = CELLLINE_V2 / "construct_sequences.tsv"
EXCLUDED_81 = CELLLINE_V2 / "excluded_untransfected_fragment.tsv"
CELLLINE_CV_PRED = CELLLINE_V2 / "model_cv_predictions.tsv"

VERSION = "v6.16.1-sequence.rc1"
BATCH = "batch21"

MPRA_LEN = 164
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
AMBIG = 4
N_FOLD = 5
FOLD_SEEDS = (20261008, 20261009, 20261010)
N_MAX_FRACTION = 0.05


class B21InputError(ValueError):
    """Typed user-input error: bad base, length, empty record, N-heavy, unknown gene."""


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
    import datetime
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def encode_uint8(seq: str, length=None, name: str = "") -> np.ndarray:
    s = (seq or "").strip().upper()
    if len(s) == 0:
        raise B21InputError("empty sequence%s" % (" for " + name if name else ""))
    if length is not None and len(s) != length:
        raise B21InputError("length %d != required %d%s"
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
    recs, name, buf = [], None, []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
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
        raise B21InputError("no FASTA records found in %s" % path)
    return recs


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
    if len(p) < 2:
        return float("nan")
    a, b = _rank(p), _rank(o)
    a = a - a.mean(); b = b - b.mean()
    d = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def mae(pred, obs) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(obs, float))))


def mse(pred, obs) -> float:
    return float(np.mean((np.asarray(pred, float) - np.asarray(obs, float)) ** 2))


def r2(pred, obs) -> float:
    o = np.asarray(obs, float)
    ss = ((o - o.mean()) ** 2).sum()
    return float(1.0 - ((np.asarray(pred, float) - o) ** 2).sum() / ss) if ss > 0 else float("nan")


def top_decile_precision(pred, obs, sha=None) -> dict:
    """k = ceil(0.1*n), descending score, ties broken by stable sequenceSHA.

    The measured top-decile is defined INDEPENDENTLY (descending measured value,
    same stable tie key). Precision = |pred_top_k intersect meas_top_k| / k.
    """
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    n = len(p)
    k = int(math.ceil(0.1 * n))
    if sha is None:
        sha = [str(i) for i in range(n)]
    key = np.array([str(s) for s in sha])
    pred_order = np.lexsort((key, -p))
    meas_order = np.lexsort((key, -o))
    pset = set(pred_order[:k].tolist())
    mset = set(meas_order[:k].tolist())
    inter = len(pset & mset)
    base = k / n
    return {"k": k, "n": n, "intersection": inter,
            "precision": inter / k, "enrichment": (inter / k) / base if base > 0 else float("nan")}


def group_bootstrap_ci(pred, obs, groups, metric=spearman, n_boot=2000, seed=20261023):
    """Bootstrap over INDEPENDENT GROUPS (not rows). Resample groups with replacement."""
    pred = np.asarray(pred, float); obs = np.asarray(obs, float)
    groups = np.asarray([str(g) for g in groups])
    ug = np.unique(groups)
    if len(ug) < 3:
        return {"point": float(metric(pred, obs)), "lo": None, "hi": None,
                "n_groups": int(len(ug)), "note": "too few independent groups for a bootstrap CI"}
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
    return {"point": float(metric(pred, obs)),
            "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_groups": int(len(ug)), "n_boot": int(len(vals))}


def all_metrics(pred, obs, sha=None, groups=None) -> dict:
    out = {"n": int(len(pred)), "spearman": spearman(pred, obs), "mae": mae(pred, obs),
           "mse": mse(pred, obs), "r2": r2(pred, obs)}
    if sha is not None:
        out["top_decile"] = top_decile_precision(pred, obs, sha)
    if groups is not None:
        out["spearman_group_bootstrap"] = group_bootstrap_ci(pred, obs, groups, spearman)
        out["mae_group_bootstrap"] = group_bootstrap_ci(pred, obs, groups, mae)
    return out
